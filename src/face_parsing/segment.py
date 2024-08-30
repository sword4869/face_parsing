#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
import os.path as osp

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from face_parsing.model import BiSeNet
from tqdm import tqdm


def vis_parsing_maps(
    im,
    parsing_anno,
    sample_name,
    save_root=None,
    color_style=None,
    stride=None,
    save_masks=None,
    masks_partition_by_name=None,
    save_parsing_anno=None,
    save_merge=None,
    save_weighted=None,
    chosen_parts=None,
    chosen_filename=None,
    chosen_reverse=None,
    **kwargs
):
    '''将分割结果可视化
    Args:
        im: 输入图片 [512, 512, 3]
        parsing_anno: 每个像素上分割结果的类别, shape=(512, 512)
        sample_name: 保存文件名

        stride: 缩放比例
        save_root: 保存路径
        color_style: 选择颜色
            face-parsing-style 为 face-parsing.PyTorch 代码的颜色,
            CelebAMask-HQ-style 为 CelebAMask-HQ的颜色 CelebAMask-HQ/face_parsing/Data_preprocessing/g_color.py
        save_masks: 是否保存每个部分的mask
        partition_by_name： True则masks/00001/各个mask, False则masks/各个mask/00001
        save_weighted: 是否保存加权叠加的图片
        save_merge: 是否保存合并后的mask
    '''
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

    #
    # if masks_partition_by_name:
    #     os.makedirs(osp.join(save_root, 'masks', sample_name), exist_ok=True)
    # else:
    #     for i in range(19):
    #         os.makedirs(osp.join(save_root, 'masks', attr[i]["name"]), exist_ok=True)

    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    # 所有融合mask（彩色）merge，指定融合mask（白色）chosen_merge
    merge = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3))
    chosen_merge = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3))

    if save_masks:
        os.makedirs(osp.join(save_root, 'masks'), exist_ok=True)
        if masks_partition_by_name:
            for i in range(19):
                os.makedirs(osp.join(save_root, 'masks', attr[i]["name"]), exist_ok=True)

    # 画出每个部分的mask，即 vis_parsing_anno_color
    for pi in np.unique(vis_parsing_anno):
        index = np.where(vis_parsing_anno == pi)
        # 每个部分：（彩色）vis_parsing_anno_color, （白色）mask
        mask = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3))
        if save_merge:
            vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3))

        # 赋值
        mask[index[0], index[1], :] = 255
        if save_merge:
            vis_parsing_anno_color[index[0], index[1], :] = attr[pi][color_style]
            merge += vis_parsing_anno_color
        if chosen_parts is not None and pi in chosen_parts:
            chosen_merge += mask

        if save_masks:
            mask_path = osp.join(save_root, 'masks', attr[pi]["name"], f'{sample_name}_{attr[pi]["name"]}.png')
            if masks_partition_by_name:
                mask_path = osp.join(save_root, 'masks', sample_name, f'{str(pi).zfill(2)}_{attr[pi]["name"]}.png')
            cv2.imwrite(mask_path, mask)

    # 保存分割结果
    if save_parsing_anno:
        os.makedirs(osp.join(save_root, 'parsing'), exist_ok=True)
        cv2.imwrite(osp.join(save_root, 'parsing', 'parsing_' + sample_name + '.png'), vis_parsing_anno)

    # 保存整体融合后的mask
    if save_merge:
        os.makedirs(osp.join(save_root, 'merge'), exist_ok=True)
        cv2.imwrite(osp.join(save_root, 'merge', 'merge_' + sample_name + '.png') , merge)

    # 加权叠加
    if save_weighted:
        os.makedirs(osp.join(save_root, 'weighted'), exist_ok=True)
        im = np.array(im).astype(np.uint8)
        vis_im = cv2.addWeighted(cv2.cvtColor(im, cv2.COLOR_RGB2BGR), 0.4, merge, 0.6, 0)
        cv2.imwrite(osp.join(save_root, 'weighted', 'weighted_' + sample_name + '.png'), vis_im)

    # 保存指定融合后的mask
    if chosen_parts is not None:
        if chosen_reverse:
            chosen_merge = 255 - chosen_merge
        os.makedirs(osp.join(save_root, chosen_filename), exist_ok=True)
        cv2.imwrite(osp.join(save_root, chosen_filename, f'{sample_name}_{chosen_filename}.png') , chosen_merge)



def evaluate(args):
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    net.load_state_dict(torch.load(args.ckpt))
    net.eval()
    print('* loaded ckpt from {}'.format(args.ckpt))

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    print('* evaluate on {}'.format(args.img_path))
    print('* save results to {}'.format(args.save_root))
    os.makedirs(args.save_root, exist_ok=True)

    with torch.no_grad():
        for img_name in tqdm(os.listdir(args.img_path)):
            image = Image.open(osp.join(args.img_path, img_name))
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            vis_parsing_maps(
                image, 
                parsing, 
                img_name[: -4],
                **vars(args)
            )

def run_cli():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='./pretrain/79999_iter.pth', help='checkpoint path')
    parser.add_argument('--img_path', type=str, default='./test_img', help='data path')
    parser.add_argument('--save_root', type=str, default='./test_res', help='results path')

    parser.add_argument('--color_style', choices=['face-parsing-style', 'CelebAMask-HQ-style'], default='face-parsing-style', help='color style')
    parser.add_argument('--stride', type=int, default=1, help='stride')

    parser.add_argument('--save_masks', action='store_true', help='save masks')
    parser.add_argument('--masks_partition_by_name', action='store_true', help='partition the masks image by name')

    parser.add_argument('--save_parsing_anno', action='store_true', help='save parsing annotation')
    parser.add_argument('--save_merge', action='store_true', help='save merge')
    parser.add_argument('--save_weighted', action='store_true', help='save weighted')
    parser.add_argument('--chosen_parts', type=int, nargs='+', default=None, help='chosen parts')
    parser.add_argument('--chosen_filename', type=str, default='chose', help='image name')
    parser.add_argument('--chosen_reverse', action='store_true', help='reverse the chosen parts')
    args = parser.parse_args()
    evaluate(args)

if __name__ == "__main__":
    run_cli()