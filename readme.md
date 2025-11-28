# Versagen (AAAI 2025 Oral)

<a href='https://arxiv.org/abs/2412.11594'><img src='https://img.shields.io/badge/technical-paper-green'></a>

This is the official implementation of "VersaGen: Unleashing Versatile Visual Control for Text-to-Image Synthesis".

<p align="center">
  <img src="./image/show.png" alt="VersaGen">
</p>

# Environmental Installation
```shell
conda create -n VersaGen python=3.10
conda activate VersaGen
pip install -r requirements.txt
```

# Download Weight
Please download the model weight and save it in the `checkpoint` folder.
```
https://pan.baidu.com/s/1Y0yEx0E4hjcVxiSCPYOwjg passward: 2ept 
```

# Run the demo
Run the following command in the terminal
```
CUDA_VISIBLE_DEVICES=5 python versagen_show.py --server_port 7899
```

Open the gradio demo in your browser
```
ip_address:7899/
```

# Acknowledgements
This project is developped on the codebase of [ControlNet](https://huggingface.co/blog/train-your-controlnet). We appreciate this great work!

# Citation
If you find this codebase useful for your research, please use the following entry.
```
@inproceedings{chen2025versagen,
  title={VersaGen: Unleashing Versatile Visual Control for Text-to-Image Synthesis},
  author={Chen, Zhipeng and Yang, Lan and Qi, Yonggang and Zhang, Honggang and Pang, Kaiyue and Li, Ke and Song, Yi-Zhe},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={3},
  pages={2394--2402},
  year={2025}
}
```



