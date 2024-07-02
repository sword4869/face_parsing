[README_ZH.md](https://github.com/sword4869/face_parsing/blob/master/README_ZH.md) | [README.md](https://github.com/sword4869/face_parsing/blob/master/README.md) 

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
    ├── merge_00000.png             # 融合mask
    ├── merge_116_ori.png
    ├── weighted_00000.png          # 叠加原图
    ├── weighted_116_ori.png
    ├── parsing_00000.png           # 分类结果，每个像素的值是[0, 18]
    ├── parsing_116_ori.png
    └── masks                       # 各部分mask
        ├── 00000
        │   ├── background.png
        │   ├── eye_g.png
        │   ├── hair.png
        │   ├── hat.png
        └── 116_ori
            ├── background.png
            ├── ear_r.png
            ├── eye_g.png
            ├── hair.png
```
```python
# 在face_parsing下
$ face_parsing

# 在face_parsing路径外
$ face_parsing --ckpt ~/79999_iter.pth --res_path ~/test_res --img_path ~/test_img
```
生成脚本为 `segment.py`。默认情况下，它将把分割图像输出到 `test_res` 文件夹中。

`masks`: 仅展示识别部分。



color1 原来 [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) 代码的颜色, 

color2 `CelebAMask-HQ`的颜色 `CelebAMask-HQ/face_parsing/Data_preprocessing/g_color.py`

| Index |    Name    |    Color1     |    Color2     |
| :---: | :--------: | :-----------: | :-----------: |
|   0   | background |   255, 0, 0   |    0, 0, 0    |
|   1   |    skin    |  255, 85, 0   |   204, 0, 0   |
|   2   |    nose    |  255, 170, 0  |  76, 153, 0   |
|   3   |   eye_g    |  255, 0, 85   |  204, 204, 0  |
|   4   |   l_eye    |  255, 0, 170  |  51, 51, 255  |
|   5   |   r_eye    |   0, 255, 0   |  204, 0, 204  |
|   6   |   l_brow   |  85, 255, 0   |  0, 255, 255  |
|   7   |   r_brow   |  170, 255, 0  | 255, 204, 204 |
|   8   |   l_ear    |  0, 255, 85   |  102, 51, 0   |
|   9   |   r_ear    |  0, 255, 170  |   255, 0, 0   |
|  10   |   mouth    |   0, 0, 255   |  102, 204, 0  |
|  11   |   u_lip    |  85, 0, 255   |  255, 255, 0  |
|  12   |   l_lip    |  170, 0, 255  |   0, 0, 153   |
|  13   |    hair    |  0, 85, 255   |   0, 0, 204   |
|  14   |    hat     |  0, 170, 255  | 255, 51, 153  |
|  15   |   ear_r    |  255, 255, 0  |  0, 204, 204  |
|  16   |   neck_l   | 255, 255, 85  |   0, 51, 0    |
|  17   |    neck    | 255, 255, 170 | 255, 153, 51  |
|  18   |   cloth    |  255, 0, 255  |   0, 204, 0   |

![image-20240702214442271](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407022144311.png)

![image-20240702214108718](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407022141752.png)

![image-20240702214100671](https://cdn.jsdelivr.net/gh/sword4869/pic1@main/images/202407022141703.png)

# 参考资料

> folk from https://github.com/zllrunning/face-parsing.PyTorch, https://github.com/dw-dengwei/face-seg

1. [Code: BiSeNet](https://github.com/CoinCheung/BiSeNet)
2. [Paper: BiSeNetV1](https://arxiv.org/abs/1808.00897)
3. [Paper: BiSeNetV2](https://arxiv.org/abs/2004.02147)
