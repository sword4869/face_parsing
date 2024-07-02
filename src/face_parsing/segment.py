#!/usr/bin/python
# -*- encoding: utf-8 -*-

from face_parsing.model import BiSeNet

import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
from tqdm import tqdm


def vis_parsing_maps(
    im,             
    parsing_anno,   
    stride,         
    save_root,      
    sample_name,    
    color_chosen=1,
    save_masks=True,
    save_weighted=True,
    save_merge=True,
    save_parsing_anno=True,
    chosen_parts=None
):
    '''将分割结果可视化
    Args:
        im: 输入图片 [512, 512, 3]
        parsing_anno: 每个像素上分割结果的类别, shape=(512, 512)
        stride: 缩放比例
        save_root: 保存路径
        sample_name: 保存文件名
        color_chosen: 选择颜色
            1为color1 原来 face-parsing.PyTorch 代码的颜色, 
            2为color2 CelebAMask-HQ的颜色 CelebAMask-HQ/face_parsing/Data_preprocessing/g_color.py
        save_masks: 是否保存每个部分的mask
        save_weighted: 是否保存加权叠加的图片
        save_merge: 是否保存合并后的mask
    '''
    os.makedirs(osp.join(save_root, 'masks', sample_name), exist_ok=True)
    

    # 19 parts
    attr = {
        0: {
            'name': 'background',
            'color1': [255, 0, 0],
            'color2': [0, 0, 0],
        },
        1: {
            'name': 'skin',
            'color1': [255, 85, 0],
            'color2': [204, 0, 0],
        },
        2: {
            'name': 'nose',
            'color1': [255, 170, 0],
            'color2': [76, 153, 0],
        },
        3: {
            'name': 'eye_g',
            'color1': [255, 0, 85],
            'color2': [204, 204, 0],
        },
        4: {
            'name': 'l_eye',
            'color1': [255, 0, 170],
            'color2': [51, 51, 255],
        },
        5: {
            'name': 'r_eye',
            'color1': [0, 255, 0],
            'color2': [204, 0, 204],
        },
        6: {
            'name': 'l_brow',
            'color1': [85, 255, 0],
            'color2': [0, 255, 255],
        },
        7: {
            'name': 'r_brow',
            'color1': [170, 255, 0],
            'color2': [255, 204, 204],
        },
        8: {
            'name': 'l_ear',
            'color1': [0, 255, 85],
            'color2': [102, 51, 0],
        },
        9: {
            'name': 'r_ear',
            'color1': [0, 255, 170],
            'color2': [255, 0, 0],
        },
        10: {
            'name': 'mouth',
            'color1': [0, 0, 255],
            'color2': [102, 204, 0],
        },
        11: {
            'name': 'u_lip',
            'color1': [85, 0, 255],
            'color2': [255, 255, 0],
        },
        12: {
            'name': 'l_lip',
            'color1': [170, 0, 255],
            'color2': [0, 0, 153],
        },
        13: {
            'name': 'hair',
            'color1': [0, 85, 255],
            'color2': [0, 0, 204],
        },
        14: {
            'name': 'hat',
            'color1': [0, 170, 255],
            'color2': [255, 51, 153],
        },
        15: {
            'name': 'ear_r',
            'color1': [255, 255, 0],
            'color2': [0, 204, 204],
        },
        16: {
            'name': 'neck_l',
            'color1': [255, 255, 85],
            'color2': [0, 51, 0],
        },
        17: {
            'name': 'neck',
            'color1': [255, 255, 170],
            'color2': [255, 153, 51],
        },
        18: {
            'name': 'cloth',
            'color1': [255, 0, 255],
            'color2': [0, 204, 0],
        },
    }


    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    merge = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3))


    # 画出每个部分的mask，即 vis_parsing_anno_color
    for pi in np.unique(vis_parsing_anno):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3))
        mask = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3))
        vis_parsing_anno_color[index[0], index[1], :] = attr[pi]['color' + str(color_chosen)]
        mask[index[0], index[1], :] = 255
        merge = merge + vis_parsing_anno_color
        if save_masks:
            cv2.imwrite(
                osp.join(save_root, 'masks', sample_name, attr[pi]['name'] + '.png'), 
                mask
            )
    merge = merge.astype(np.uint8)
    if merge.max() > 255:
        raise ValueError('Color value out of range, please check the color value of each part.')
    
    # 加权叠加
    if save_weighted:
        im = np.array(im).astype(np.uint8)
        vis_im = cv2.addWeighted(cv2.cvtColor(im, cv2.COLOR_RGB2BGR), 0.4, merge, 0.6, 0)
        cv2.imwrite(osp.join(save_root, 'weighted_' + sample_name + '.png'), vis_im)

    # 保存合并后的mask
    if save_merge:
        cv2.imwrite(osp.join(save_root, 'merge_' + sample_name + '.png') , merge)

    # 保存分割结果
    if save_parsing_anno:
        cv2.imwrite(osp.join(save_root, 'parsing_' + sample_name + '.png'), vis_parsing_anno)


def evaluate(img_path, res_path, ckpt):
    '''
    Args:
        img_path: 数据路径
        res_path: 输出保存路径
        ckpt: 模型路径
    '''

    if not os.path.exists(res_path):
        os.makedirs(res_path)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    net.load_state_dict(torch.load(ckpt))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        for img_name in tqdm(os.listdir(img_path)):
            image = Image.open(osp.join(img_path, img_name))
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            vis_parsing_maps(
                image, 
                parsing, 
                stride=1, 
                save_root=res_path,
                sample_name=img_name[: -4]
            )

def run_cli():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--res_path', type=str, default='./test_res', help='results path')
    parser.add_argument('--img_path', type=str, default='./test_img', help='data path')
    parser.add_argument('--ckpt', type=str, default='./pretrain/79999_iter.pth', help='checkpoint path')
    args = parser.parse_args()
    evaluate(args.img_path, args.res_path, args.ckpt)

if __name__ == "__main__":
    run_cli()