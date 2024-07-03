#!/usr/bin/python
# -*- encoding: utf-8 -*-

from ast import arg
from calendar import c
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
    color_style='face-parsing-style',
    save_masks=True,
    save_weighted=True,
    save_merge=True,
    save_parsing_anno=True,
    chosen_parts=None,
    reverse=False
):
    '''将分割结果可视化
    Args:
        im: 输入图片 [512, 512, 3]
        parsing_anno: 每个像素上分割结果的类别, shape=(512, 512)
        stride: 缩放比例
        save_root: 保存路径
        sample_name: 保存文件名
        color_style: 选择颜色
            face-parsing-style 为 face-parsing.PyTorch 代码的颜色, 
            CelebAMask-HQ-style 为 CelebAMask-HQ的颜色 CelebAMask-HQ/face_parsing/Data_preprocessing/g_color.py
        save_masks: 是否保存每个部分的mask
        save_weighted: 是否保存加权叠加的图片
        save_merge: 是否保存合并后的mask
    '''
    os.makedirs(osp.join(save_root, 'masks', sample_name), exist_ok=True)
    

    # 19 parts
    attr = {
        0: {
            'name': 'background',
            'face-parsing-style': [255, 0, 0],
            'CelebAMask-HQ-style': [0, 0, 0],
        },
        1: {
            'name': 'skin',
            'face-parsing-style': [255, 85, 0],
            'CelebAMask-HQ-style': [204, 0, 0],
        },
        2: {
            'name': 'l_brow',
            'face-parsing-style': [255, 170, 0],
            'CelebAMask-HQ-style': [0, 255, 255],
        },
        3: {
            'name': 'r_brow',
            'face-parsing-style': [255, 0, 85],
            'CelebAMask-HQ-style': [255, 204, 204],
        },
        4: {
            'name': 'l_eye',
            'face-parsing-style': [255, 0, 170],
            'CelebAMask-HQ-style': [51, 51, 255],
        },
        5: {
            'name': 'r_eye',
            'face-parsing-style': [0, 255, 0],
            'CelebAMask-HQ-style': [204, 0, 204],
        },
        6: {
            'name': 'eye_g',
            'face-parsing-style': [85, 255, 0],
            'CelebAMask-HQ-style': [204, 204, 0],
        },
        7: {
            'name': 'l_ear',
            'face-parsing-style': [170, 255, 0],
            'CelebAMask-HQ-style': [102, 51, 0],
        },
        8: {
            'name': 'r_ear',
            'face-parsing-style': [0, 255, 85],
            'CelebAMask-HQ-style': [255, 0, 0],
        },
        9: {
            'name': 'ear_r',
            'face-parsing-style': [0, 255, 170],
            'CelebAMask-HQ-style': [0, 204, 204],
        },
        10: {
            'name': 'nose',
            'face-parsing-style': [0, 0, 255],
            'CelebAMask-HQ-style': [76, 153, 0],
        },
        11: {
            'name': 'mouth',
            'face-parsing-style': [85, 0, 255],
            'CelebAMask-HQ-style': [102, 204, 0],
        },
        12: {
            'name': 'u_lip',
            'face-parsing-style': [170, 0, 255],
            'CelebAMask-HQ-style': [255, 255, 0],
        },
        13: {
            'name': 'l_lip',
            'face-parsing-style': [0, 85, 255],
            'CelebAMask-HQ-style': [0, 0, 153],
        },
        14: {
            'name': 'neck',
            'face-parsing-style': [0, 170, 255],
            'CelebAMask-HQ-style': [255, 153, 51],
        },
        15: {
            'name': 'neck_l',
            'face-parsing-style': [255, 255, 0],
            'CelebAMask-HQ-style': [0, 51, 0],
        },
        16: {
            'name': 'cloth',
            'face-parsing-style': [255, 255, 85],
            'CelebAMask-HQ-style': [0, 204, 0],
        },
        17: {
            'name': 'hair',
            'face-parsing-style': [255, 255, 170],
            'CelebAMask-HQ-style': [0, 0, 204],
        },
        18: {
            'name': 'hat',
            'face-parsing-style': [255, 0, 255],
            'CelebAMask-HQ-style': [255, 51, 153],
        },
    }


    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    # 所有融合mask（彩色）merge，指定融合mask（白色）chosen_merge
    merge = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3))
    chosen_merge = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3))


    # 画出每个部分的mask，即 vis_parsing_anno_color
    for pi in np.unique(vis_parsing_anno):
        index = np.where(vis_parsing_anno == pi)
        # 每个部分：（彩色）vis_parsing_anno_color, （白色）mask
        vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3))
        mask = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3))
        
        # 赋值
        vis_parsing_anno_color[index[0], index[1], :] = attr[pi][color_style]
        mask[index[0], index[1], :] = 255
        merge += vis_parsing_anno_color
        if chosen_parts is not None and pi in chosen_parts:
            chosen_merge += mask

        if save_masks:
            cv2.imwrite(
                osp.join(save_root, 'masks', sample_name, f'{str(pi).zfill(2)}_{attr[pi]["name"]}.png'), 
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

    # 保存整体融合后的mask
    if save_merge:
        cv2.imwrite(osp.join(save_root, 'merge_' + sample_name + '.png') , merge)

    # 保存指定融合后的mask
    if chosen_parts is not None:
        if reverse:
            chosen_merge = 255 - chosen_merge
            cv2.imwrite(osp.join(save_root, 'chosen_merge_' + sample_name + '.png') , chosen_merge)

    # 保存分割结果
    if save_parsing_anno:
        cv2.imwrite(osp.join(save_root, 'parsing_' + sample_name + '.png'), vis_parsing_anno)


def evaluate(img_path, res_path, ckpt, chosen_parts, reverse, color_style):
    '''
    Args:
        img_path: 数据路径
        res_path: 输出保存路径
        ckpt: 模型路径
        chosen_parts: 选择的部分
    '''

    if not os.path.exists(res_path):
        os.makedirs(res_path)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    net.load_state_dict(torch.load(ckpt))
    net.eval()
    print('* loaded ckpt from {}'.format(ckpt))

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
                sample_name=img_name[: -4],
                chosen_parts=chosen_parts,
                reverse=reverse,
                color_style=color_style
            )

def run_cli():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--res_path', type=str, default='./test_res', help='results path')
    parser.add_argument('--img_path', type=str, default='./test_img', help='data path')
    parser.add_argument('--ckpt', type=str, default='./pretrain/79999_iter.pth', help='checkpoint path')
    parser.add_argument('--chosen_parts', type=int, nargs='+', default=None, help='chosen parts')
    parser.add_argument('--reverse', action='store_true', help='reverse the chosen parts')
    parser.add_argument('--color_style', choices=['face-parsing-style', 'CelebAMask-HQ-style'], default='face-parsing-style', help='color style')
    args = parser.parse_args()
    evaluate(args.img_path, args.res_path, args.ckpt, args.chosen_parts, args.reverse, args.color_style)

if __name__ == "__main__":
    run_cli()