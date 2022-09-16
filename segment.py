#!/usr/bin/python
# -*- encoding: utf-8 -*-

from model import BiSeNet

import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2


def vis_parsing_maps(
    im,
    parsing_anno,
    stride,
    save_root,
    sample_name,
    save_im=False,
):
    if not os.path.exists(osp.join(save_root, 'unmerge', sample_name)):
        os.makedirs(osp.join(save_root, 'unmerge', sample_name))

    atts = ['background', 'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 
            'ear_r', 'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']

    # Colors specific parts (not zero)
    part_colors = [
        [0x00] * 3, # bg
        [0x64] * 3, # skin
        [0xa9] * 3, # l brow
        [0xa9] * 3, # r brow
        [0xfe] * 3, # l eye
        [0xfe] * 3, # r eye
        [0x00] * 3, # eye g
        [0x00] * 3, # l ear
        [0x00] * 3, # r ear
        [0x00] * 3, # ear r
        [0xff] * 3, # nose
        [0xaa] * 3, # mouth
        [0xaa] * 3, # u lip
        [0xaa] * 3, # l lip
        [0x00] * 3, # neck
        [0x00] * 3, # neck l
        [0x00] * 3, # cloth
        [0x00] * 3, # hair
        [0x00] * 3, # hat
    ]

    im = np.array(im)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    merge = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3))

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        if part_colors[pi] == [0, 0, 0]:
            continue
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3))
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]
        merge = merge + vis_parsing_anno_color
        if save_im:
            cv2.imwrite(
                osp.join(save_root, 'unmerge', sample_name, atts[pi] + '.png'), 
                vis_parsing_anno_color
            )

    if save_im:
        cv2.imwrite(osp.join(save_root, 'Mask_' + sample_name + '.png') , merge)


def evaluate(respth='test_res', dspth='./data', cp='model_final_diss.pth'):

    if not os.path.exists(respth):
        os.makedirs(respth)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = osp.join('.', cp)
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        for image_path in os.listdir(dspth):
            img = Image.open(osp.join(dspth, image_path))
            # image = img.resize((512, 512), Image.BILINEAR)
            img = to_tensor(img)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)

            vis_parsing_maps(
                img.cpu(), 
                parsing, 
                stride=1, 
                save_im=True, 
                save_root=respth,
                sample_name=image_path[: -4]
            )


if __name__ == "__main__":
    evaluate(dspth='test_img/', cp='pretrain/checkpoint.pth')