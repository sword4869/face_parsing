[README_ZH.md](https://github.com/sword4869/face_parsing/blob/master/README.md) | [README_EN.md](https://github.com/sword4869/face_parsing/blob/master/README_EN.md) 

# 项目描述 

该仓库用于生成人脸图像的语义分割。

# 安装

下载 https://drive.google.com/open?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812 的权重文件 79999_iter.pth
```python
# pip install -e git+https://github.com/sword4869/face_parsing.git#egg=face_parsing
pip install face_parsing
```

# 输入和输出
```
├── pretrain
│   └── 79999_iter.pth      # ckpt
├── test_img                # 输入
│   ├── 00000.jpg
│   └── 116_ori.png
└── test_res                # 输出
    ├── chosen_merge_00000.png      # 指定部分融合mask
    ├── chosen_merge_116_ori.png
    ├── merge_00000.png             # 融合mask
    ├── merge_116_ori.png
    ├── weighted_00000.png          # 叠加原图
    ├── weighted_116_ori.png
    ├── parsing_00000.png           # 分类结果，每个像素的值是[0, 18]
    ├── parsing_116_ori.png
    └── masks                       # 各部分mask
        ├── 00000
        │   ├── 00_background.png
        │   ├── 01_skin.png
        │   ├── 02_l_brow.png
        │   ├── 03_r_brow.png
        └── 116_ori
            ├── 00_background.png
            ├── 01_skin.png
            ├── 02_l_brow.png
            ├── 03_r_brow.png
```
```bash
usage: face_parsing [-h] [--res_path RES_PATH] [--img_path IMG_PATH] [--ckpt CKPT] [--chosen_parts CHOSEN_PARTS [CHOSEN_PARTS ...]] [--reverse] [--color_style {face-parsing-style,CelebAMask-HQ-style}]

options:
  -h, --help            show this help message and exit
  --res_path RES_PATH   results path
  --img_path IMG_PATH   data path
  --ckpt CKPT           checkpoint path
  --chosen_parts CHOSEN_PARTS [CHOSEN_PARTS ...]
                        chosen parts
  --reverse             reverse the chosen parts
  --color_style {face-parsing-style,CelebAMask-HQ-style}
                        color style

# 在face_parsing下
$ face_parsing

# 在face_parsing路径外
$ face_parsing --ckpt ~/79999_iter.pth --res_path ~/test_res --img_path ~/test_img
```

| Index |   Name   | [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) Style RGB | [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ/blob/master/face_parsing/README.md) Style RGB |
|-------|----------|------------------------|-------------------------|
|   0   | background | [255, 0, 0]          | [0, 0, 0]               |
|   1   | skin     | [255, 85, 0]          | [204, 0, 0]             |
|   2   | l_brow   | [255, 170, 0]         | [0, 255, 255]           |
|   3   | r_brow   | [255, 0, 85]          | [255, 204, 204]         |
|   4   | l_eye    | [255, 0, 170]         | [51, 51, 255]           |
|   5   | r_eye    | [0, 255, 0]           | [204, 0, 204]           |
|   6   | eye_g    | [85, 255, 0]          | [204, 204, 0]           |
|   7   | l_ear    | [170, 255, 0]         | [102, 51, 0]            |
|   8   | r_ear    | [0, 255, 85]          | [255, 0, 0]             |
|   9   | ear_r    | [0, 255, 170]         | [0, 204, 204]           |
|   10  | nose     | [0, 0, 255]           | [76, 153, 0]            |
|   11  | mouth    | [85, 0, 255]          | [102, 204, 0]           |
|   12  | u_lip    | [170, 0, 255]         | [255, 255, 0]           |
|   13  | l_lip    | [0, 85, 255]          | [0, 0, 153]             |
|   14  | neck     | [0, 170, 255]         | [255, 153, 51]          |
|   15  | neck_l   | [255, 255, 0]         | [0, 51, 0]              |
|   16  | cloth    | [255, 255, 85]        | [0, 204, 0]             |
|   17  | hair     | [255, 255, 170]       | [0, 0, 204]             |
|   18  | hat      | [255, 0, 255]         | [255, 51, 153]          |

## face-parsing的模型效果

![image-20240703160744779](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407031607840.png)

![image-20240703160815874](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407031608907.png)

![image-20240703160828697](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407031608737.png)

# 参考资料

> folk from https://github.com/zllrunning/face-parsing.PyTorch, https://github.com/dw-dengwei/face-seg

1. [Code: BiSeNet](https://github.com/CoinCheung/BiSeNet)
2. [Paper: BiSeNetV1](https://arxiv.org/abs/1808.00897)
3. [Paper: BiSeNetV2](https://arxiv.org/abs/2004.02147)
