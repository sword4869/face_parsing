# 项目描述 

基于 [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch), 该仓库用于生成人脸图像的语义分割。

# 安装

下载 https://drive.google.com/open?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812 的权重文件 79999_iter.pth

```bash
pip install git+https://github.com/sword4869/face_parsing.git#egg=face_parsing
```

# 输入和输出
```
├── pretrain
│   └── 79999_iter.pth      # ckpt
```
```bash
usage: face_parsing [-h] [--ckpt CKPT] [--img_path IMG_PATH] [--save_root SAVE_ROOT] [--color_style {face-parsing-style,CelebAMask-HQ-style}] [--stride STRIDE] [--save_masks] [--masks_partition_by_name]
                    [--save_parsing_anno] [--save_merge] [--save_weighted] [--chosen_parts CHOSEN_PARTS [CHOSEN_PARTS ...]] [--chosen_filename CHOSEN_FILENAME] [--chosen_reverse]

optional arguments:
  -h, --help            show this help message and exit
  --ckpt CKPT           checkpoint path
  --img_path IMG_PATH   data path
  --save_root SAVE_ROOT
                        results path
  --color_style {face-parsing-style,CelebAMask-HQ-style}
                        color style
  --stride STRIDE       stride
  --save_masks          save masks
  --masks_partition_by_name
                        partition the masks image by name
  --save_parsing_anno   save parsing annotation
  --save_merge          save merge
  --save_weighted       save weighted
  --chosen_parts CHOSEN_PARTS [CHOSEN_PARTS ...]
                        chosen parts
  --chosen_filename CHOSEN_FILENAME
                        image name
  --chosen_reverse      reverse the chosen parts

# 在face_parsing下
$ face_parsing

# 在face_parsing路径外
$ face_parsing --ckpt ~/79999_iter.pth
```

# chosen_parts

```bash
# neck和head前景（反选背景和cloth）
$ subject='bala'
$ face_parsing  --ckpt /home/lab/Documents/face-seg/pretrain/79999_iter.pth \
    --img_path /media/lab/新加卷/DataSet/FlashAvatar/flash/dataset/$subject/imgs \
    --save_root /media/lab/新加卷/DataSet/FlashAvatar/flash/dataset/$subject/parsing \
	--chosen_parts 0 16 --chosen_filename neckhead --chosen_reverse

# mouth u_lip 和 lip
$ face_parsing  --ckpt /home/lab/Documents/face-seg/pretrain/79999_iter.pth \
    --img_path /media/lab/新加卷/DataSet/FlashAvatar/flash/dataset/$subject/imgs \
    --save_root /media/lab/新加卷/DataSet/FlashAvatar/flash/dataset/$subject/parsing \
	--chosen_parts 11 12 13 --chosen_filename mouth
```

| Index |   Name   | [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) Style RGB | [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ/blob/master/face_parsing/README.md) Style RGB | 备注  |
|-------|----------|------------------------|-------------------------|-----|
|   0   | background | [255, 0, 0]          | [0, 0, 0]               |     |
|   1   | skin     | [255, 85, 0]          | [204, 0, 0]             |     |
|   2   | l_brow   | [255, 170, 0]         | [0, 255, 255]           |     |
|   3   | r_brow   | [255, 0, 85]          | [255, 204, 204]         |     |
|   4   | l_eye    | [255, 0, 170]         | [51, 51, 255]           |     |
|   5   | r_eye    | [0, 255, 0]           | [204, 0, 204]           |     |
|   6   | eye_g    | [85, 255, 0]          | [204, 204, 0]           |     |
|   7   | l_ear    | [170, 255, 0]         | [102, 51, 0]            |     |
|   8   | r_ear    | [0, 255, 85]          | [255, 0, 0]             |     |
|   9   | ear_r    | [0, 255, 170]         | [0, 204, 204]           |     |
|   10  | nose     | [0, 0, 255]           | [76, 153, 0]            |     |
|   11  | mouth    | [85, 0, 255]          | [102, 204, 0]           | 嘴内  |
|   12  | u_lip    | [170, 0, 255]         | [255, 255, 0]           | 上嘴唇 |
|   13  | l_lip    | [0, 85, 255]          | [0, 0, 153]             | 下嘴唇 |
|   14  | neck     | [0, 170, 255]         | [255, 153, 51]          |     |
|   15  | neck_l   | [255, 255, 0]         | [0, 51, 0]              | 项链  |
|   16  | cloth    | [255, 255, 85]        | [0, 204, 0]             |     |
|   17  | hair     | [255, 255, 170]       | [0, 0, 204]             |     |
|   18  | hat      | [255, 0, 255]         | [255, 51, 153]          |     |