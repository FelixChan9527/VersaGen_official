from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from diffusers.models.controlnet import ControlNetModel, ControlNetOutput
import torch.nn as nn
import torch.nn.functional as F


class ResnetBlock(nn.Module):

    def __init__(self, in_c, out_c, ksize=3, sk=False):
        super().__init__()
        ps = ksize // 2
        if in_c != out_c or sk == False:
            self.in_conv = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.in_conv = None
        self.block1 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(out_c, out_c, ksize, 1, ps)
        if sk == False:
            self.skep = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.skep = None

    def forward(self, x):
        if self.in_conv is not None:  # edit
            x = self.in_conv(x)

        h = self.block1(x)
        h = self.act(h)
        h = self.block2(h)
        if self.skep is not None:
            return h + self.skep(x)
        else:
            return h + x

class ControlNet(ControlNetModel):
    def __init__(
        self, 
        in_channels: int = 4,
        conditioning_channels: int = 3,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str, ...] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 1280,
        transformer_layers_per_block: Union[int, Tuple[int, ...]] = 1,
        encoder_hid_dim: Optional[int] = None,
        encoder_hid_dim_type: Optional[str] = None,
        attention_head_dim: Union[int, Tuple[int, ...]] = 8,
        num_attention_heads: Optional[Union[int, Tuple[int, ...]]] = None,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        addition_embed_type: Optional[str] = None,
        addition_time_embed_dim: Optional[int] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        projection_class_embeddings_input_dim: Optional[int] = None,
        controlnet_conditioning_channel_order: str = "rgb",
        conditioning_embedding_out_channels: Optional[Tuple[int, ...]] = (16, 32, 96, 256),
        global_pool_conditions: bool = False,
        addition_embed_type_num_heads: int = 64,
        input_token_channals: int = 1024,
        ):
        super(ControlNet, self).__init__(
            in_channels, 
            conditioning_channels, 
            flip_sin_to_cos, 
            freq_shift, 
            down_block_types, 
            mid_block_type, 
            only_cross_attention, 
            block_out_channels, 
            layers_per_block, 
            downsample_padding, 
            mid_block_scale_factor, 
            act_fn, 
            norm_num_groups,
            norm_eps, 
            cross_attention_dim, 
            transformer_layers_per_block, 
            encoder_hid_dim, 
            encoder_hid_dim_type, 
            attention_head_dim, 
            num_attention_heads, 
            use_linear_projection, 
            class_embed_type, 
            addition_embed_type, 
            addition_time_embed_dim, 
            num_class_embeds, 
            upcast_attention, 
            resnet_time_scale_shift, 
            projection_class_embeddings_input_dim, 
            controlnet_conditioning_channel_order, 
            conditioning_embedding_out_channels, 
            global_pool_conditions, 
            addition_embed_type_num_heads)
        
        self.fusion_layer = nn.Sequential(
                                            ResnetBlock(block_out_channels[0]*3, block_out_channels[0], sk=True),
                                            ResnetBlock(block_out_channels[0], block_out_channels[0]),
                                            ResnetBlock(block_out_channels[0], block_out_channels[0])
            )
    
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: torch.FloatTensor,
        conditioning_scale: float = 1.0,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guess_mode: bool = False,
        return_dict: bool = True,
        mask_all_obj = None,
    ) -> Union[ControlNetOutput, Tuple[Tuple[torch.FloatTensor, ...], torch.FloatTensor]]:

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)
        aug_emb = None

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        if self.config.addition_embed_type is not None:
            if self.config.addition_embed_type == "text":
                aug_emb = self.add_embedding(encoder_hidden_states)

            elif self.config.addition_embed_type == "text_time":
                if "text_embeds" not in added_cond_kwargs:
                    raise ValueError(
                        f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`"
                    )
                text_embeds = added_cond_kwargs.get("text_embeds")
                if "time_ids" not in added_cond_kwargs:
                    raise ValueError(
                        f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
                    )
                time_ids = added_cond_kwargs.get("time_ids")
                time_embeds = self.add_time_proj(time_ids.flatten())
                time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))

                add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
                add_embeds = add_embeds.to(emb.dtype)
                aug_emb = self.add_embedding(add_embeds)

        emb = emb + aug_emb if aug_emb is not None else emb

        # 2. pre-process
        sample = self.conv_in(sample)   # 噪声的输入编码

        # controlnet_cond大小 [b, 5, 3, h, w] 
        b, n, c, h, w = controlnet_cond.shape
        controlnet_cond = controlnet_cond.view((b*n, c, h, w))
        controlnet_cond = self.controlnet_cond_embedding(controlnet_cond)   # 图像编码，使之与噪声特征大小对应
        
        _, c, h, w = controlnet_cond.shape
        controlnet_cond = controlnet_cond.view((b, n*c, h, w))  # [b, n, c, h, w]
        # controlnet_cond = controlnet_cond.sum(dim=1)  # [b, c, h, w]
        controlnet_cond = self.fusion_layer(controlnet_cond)    # 不同object信息融合
        
        sample = sample + controlnet_cond   # 直接相加 后期通过这个权重进行调整

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        if self.mid_block is not None:
            if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample = self.mid_block(sample, emb)

        # 5. Control net blocks
        """
        这一步与零卷积运算，也相当于增加一个输出层.
        大小不发生改变，相当于controlnet最后一次运算
        """

        controlnet_down_block_res_samples = ()

        for down_block_res_sample, controlnet_block in zip(down_block_res_samples, self.controlnet_down_blocks):
            down_block_res_sample = controlnet_block(down_block_res_sample)
            
            # 与mask相乘，只针对局部区域作用
            if mask_all_obj is not None:
                mask_all_obj_resize = mask_all_obj.detach()   # [b, img_h, img_w]
                h_f, w_f = down_block_res_sample.shape[2:]
                mask_all_obj_resize = F.interpolate(
                                                    mask_all_obj_resize.unsqueeze(1).to(dtype=torch.float32),
                                                    size=(h_f, w_f),
                                                    mode='nearest',
                                                )           # [b, 1, h, w]
            
                down_block_res_sample = down_block_res_sample * mask_all_obj_resize
                
            controlnet_down_block_res_samples = controlnet_down_block_res_samples + (down_block_res_sample,)

        down_block_res_samples = controlnet_down_block_res_samples

        mid_block_res_sample = self.controlnet_mid_block(sample)
        # 与mask相乘，只针对局部区域作用
        if mask_all_obj is not None:
            mask_all_obj_resize = mask_all_obj.detach()   # [b, img_h, img_w]
            h_f, w_f = mid_block_res_sample.shape[2:]
            mask_all_obj_resize = F.interpolate(
                                                mask_all_obj_resize.unsqueeze(1).to(dtype=torch.float32),
                                                size=(h_f, w_f),
                                                mode='nearest',
                                            )           # [b, 1, h, w]

            mid_block_res_sample = mid_block_res_sample * mask_all_obj_resize

        # 6. scaling
        if guess_mode and not self.config.global_pool_conditions:
            scales = torch.logspace(-1, 0, len(down_block_res_samples) + 1, device=sample.device)  # 0.1 to 1.0
            scales = scales * conditioning_scale
            down_block_res_samples = [sample * scale for sample, scale in zip(down_block_res_samples, scales)]
            mid_block_res_sample = mid_block_res_sample * scales[-1]  # last one
        else:
            """
            默认进入这个
            每个分辨率的特征分别与scale相乘
            """
            down_block_res_samples = [sample * conditioning_scale for sample in down_block_res_samples]
            mid_block_res_sample = mid_block_res_sample * conditioning_scale

        if self.config.global_pool_conditions:
            """
            不进入
            将feature map进行求平均作为embedding
            """
            down_block_res_samples = [
                torch.mean(sample, dim=(2, 3), keepdim=True) for sample in down_block_res_samples
            ]
            mid_block_res_sample = torch.mean(mid_block_res_sample, dim=(2, 3), keepdim=True)

        if not return_dict:
            return (down_block_res_samples, mid_block_res_sample)

        return ControlNetOutput(
            down_block_res_samples=down_block_res_samples, mid_block_res_sample=mid_block_res_sample
        )

