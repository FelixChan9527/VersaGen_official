import math
import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import (
    Attention,
    AttnProcessor2_0,
)
import math

############################################################################################
############################################################################################

class Attention2Mask(AttnProcessor2_0):
    def __init__(self, unet, image_size=(512, 512)):
        super().__init__()
        self.atten_names = []
        # self.attn_maps = {}
        self.attn_maps = []
        self.image_size = image_size
        self.unet = self.register_cross_attention_hook(unet)
    
    def clean(self):
        # self.attn_maps = {}
        self.attn_maps = []
    
    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale=1.0,
    ):
        """
        该函数用于对stable diffusion的attention运算部分进行重写
        本质上在推理时会调用，而不是重新计算
        默认进入此处，计算更快
        """
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, scale=scale)

        if encoder_hidden_states is None:   # 如果为空，则为自注意力机制，则赋值为hidden_states
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:               # 一般跳过
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, scale=scale)
        value = attn.to_v(encoder_hidden_states, scale=scale)

        """
        此处将不同的头弄到batch维度
        """
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # 计算attention 分数
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        ####################################################################################################
        # (20,4096,77) or (40,1024,77)
        if hasattr(self, "store_attn_map"):
            # 这个判断只取cross attention map
            # self.attn_map也是此处新添加的
            self.attn_map = attention_probs
            self.attn_map = self.attn_map.reshape([batch_size, -1,          # [bs, 8, dim, 77]
                                self.attn_map.shape[-2], self.attn_map.shape[-1]])
            
        ####################################################################################################
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, scale=scale)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
    
    def scaled_dot_product_attention(self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
        # Efficient implementation equivalent to the following:
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query.dtype)
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask
        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias.to(attn_weight.device)
        attn_weight = torch.softmax(attn_weight, dim=-1)

        return torch.dropout(attn_weight, dropout_p, train=True) @ value, attn_weight
    
    def hook_fn(self, name):
        def forward_hook(module, input, output):
            global current_step
            if hasattr(module.processor, "attn_map"):
                # 前面已为self.attn_map幅值
                """
                在此处存储cross attention，会在每个step附上新值
                """ 
                # if name == atten_names[0]:  # 表示进入了一个step
                #     current_step += 1
                # 只对目标的step进行attention map取样 并且只取up block
                # 实验证明，middle block和down block都不准确
                # if name.startswith("up"):
                #     self.attn_maps[name] = module.processor.attn_map
                if name.startswith("up") and module.processor.attn_map.shape[2] == 576:   # 只取最小attn
                    # self.attn_maps[name] = module.processor.attn_map    # 在此处存储cross attention map
                    self.attn_maps.append(module.processor.attn_map)
                del module.processor.attn_map

        return forward_hook
    
    """
    将attention 层进行挂钩
    """
    def register_cross_attention_hook(self, unet):
        for name, module in unet.named_modules():
            if not name.split('.')[-1].startswith('attn2'):
                continue
            module.processor = self        # 将每个module的attention processor换成本方法
            module.processor.store_attn_map = True
            self.atten_names.append(name)
            hook = module.register_forward_hook(self.hook_fn(name))
        
        return unet
    
    def upscale(self, attn_map, target_size):
        attn_map = torch.mean(attn_map, dim=0) # (10, 32*32, 77) -> (32*32, 77)     # head维度取平均
        attn_map = attn_map.permute(1,0) # (32*32, 77) -> (77, 32*32)

        if target_size[0]*target_size[1] != attn_map.shape[1]:
            temp_size = (target_size[0]//2, target_size[1]//2)
            attn_map = attn_map.view(attn_map.shape[0], *temp_size) # (77, 32,32)
            attn_map = attn_map.unsqueeze(0) # (77,32,32) -> (1,77,32,32)

            attn_map = F.interpolate(
                attn_map.to(dtype=torch.float32),
                size=target_size,
                mode='bilinear',
                align_corners=False
            ).squeeze() # (77,64,64)
        else:
            attn_map = attn_map.to(dtype=torch.float32) # (77,64,64)

        attn_map = torch.softmax(attn_map, dim=0)
        attn_map = attn_map.reshape(attn_map.shape[0],-1) # (77,64*64)

        return attn_map
    
    def get_net_attn_map(self):

        self.attn_maps = torch.cat(self.attn_maps, dim=1)     # [bs, layers*heads, dim, 77]
        self.attn_maps = torch.mean(self.attn_maps, dim=1) # (bs, dim, 77)
        self.attn_maps = F.normalize(self.attn_maps, p=2, dim=-1)   # 有用
        self.attn_maps = self.attn_maps.permute(0, 2, 1) # (bs, 77, dim)
        self.attn_maps = self.attn_maps.reshape(                # [bs, 77, 16, 16]
                            self.attn_maps.shape[0], -1, 
                            int(math.sqrt(self.attn_maps.shape[-1])), 
                            int(math.sqrt(self.attn_maps.shape[-1]))) 
        
        bs = len(self.attn_maps) // 2
        self.attn_maps = self.attn_maps[bs:]    # 平均分块
        
        return self.attn_maps
    

    