import torch
from PIL import Image
import open_clip
import os
import json
import glob
import numpy as np
import argparse
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--img_folder', type=str, help='learning rate')
parser.add_argument('--data_root', type=str, default = '../dataset/places365', help='folder of the original dataset')
parser.add_argument('--mask_ratio', default=0.5, type=float, help='masking ratio w.r.t. one dimension')
args = parser.parse_args()


def run_clip_multi(imglist, model, preprocess):
    inputx=[]
    for imgpath in imglist:
        image = preprocess(Image.open(imgpath).convert('RGB')).unsqueeze(0)
        inputx.append(image)
    inputx = torch.cat(inputx, dim=0)
    with torch.no_grad():
        inputx = inputx.cuda()
        image_features = model(inputx)
        image_features_norm = image_features/image_features.norm(dim=-1, keepdim=True)
    return image_features_norm.squeeze().detach().cpu()


def clip_subset_img_1k(img_folder, args):
    if not os.path.isdir('./ckpt/original'):
        clip_subset_img_1k_base(args)
    if not os.path.isfile(os.path.join(img_folder, 'forget_clip_norm.txt')) or not os.path.isfile(os.path.join(img_folder, 'retain_clip_norm.txt')):
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='pretrained/open_clip_vit_h_14_laion2b_s32b_b79k.bin')
        model = model.visual
        model.cuda()
        model.eval()
        stepsize = 500

    class_list = open('datasets/place365/flist/class.txt').read().splitlines()
    retain_class = class_list[:50]
    forget_class = class_list[50:100]
    retain_imgs=[]
    for retainname in retain_class:
        tmp = glob.glob(os.path.join(args.img_folder, f'results/test/0/{args.mask_ratio:.2f}/Out'+retainname.replace('/', '_')+'_Places365*.jpg'))
        retain_imgs += tmp
    retain_imgs = sorted(retain_imgs)
    
    forget_imgs=[]
    for forgetname in forget_class:
        tmp = glob.glob(os.path.join(args.img_folder, f'results/test/0/{args.mask_ratio:.2f}/Out'+forgetname.replace('/', '_')+'_Places365*.jpg'))
        forget_imgs += tmp
    forget_imgs = sorted(forget_imgs)

    if not os.path.isfile(os.path.join(img_folder, 'forget_clip_norm.txt')):
        norm_list = []
        for i in range(0, len(forget_imgs), stepsize):
            print(i ,min(i+stepsize, len(forget_imgs)))
            imgs_batch = forget_imgs[i:min(i+stepsize, len(forget_imgs))]
            latent_norm = run_clip_multi(imgs_batch, model, preprocess)
            norm_list.append(latent_norm)  
        norm_list = torch.cat(norm_list, dim=0).numpy()            
        np.savetxt(os.path.join(img_folder, 'forget_clip_norm.txt'), np.array(norm_list))

    if not os.path.isfile(os.path.join(img_folder, 'retain_clip_norm.txt')):
        norm_list = []
        for i in range(0, len(retain_imgs), stepsize):
            print(i ,min(i+stepsize, len(retain_imgs)))
            imgs_batch = retain_imgs[i:min(i+stepsize, len(retain_imgs))]
            latent_norm = run_clip_multi(imgs_batch, model, preprocess)
            norm_list.append(latent_norm)  
        norm_list = torch.cat(norm_list, dim=0).numpy()            
        norm_list = np.array(norm_list)
        np.savetxt(os.path.join(img_folder, 'retain_clip_norm.txt'), norm_list)

    forget_norm = np.loadtxt(os.path.join(img_folder, 'forget_clip_norm.txt'))
    retain_norm = np.loadtxt(os.path.join(img_folder, 'retain_clip_norm.txt'))
    base_forget = np.zeros_like(forget_norm)
    base_retain = np.zeros_like(retain_norm)

    base_img_folder = 'ckpt/original'
    for imgid, imgname in enumerate(forget_imgs):
        base_name = os.path.basename(imgname)
        base_forget[imgid] = np.loadtxt(os.path.join(base_img_folder, base_name.replace('.jpg', '_clip_norm.txt').replace('Out_', '')))
    for imgid, imgname in enumerate(retain_imgs):
        base_name = os.path.basename(imgname)
        base_retain[imgid] = np.loadtxt(os.path.join(base_img_folder, base_name.replace('.jpg', '_clip_norm.txt').replace('Out_', '')))

    tmp = np.sum(retain_norm*base_retain, axis=1)
    cosine0 = np.mean(tmp)
    tmp = np.sum(forget_norm*base_forget, axis=1)
    cosine1 = np.mean(tmp)
    print(args.img_folder, cosine0, cosine1, sep='\t', file= open('clip_cosine.csv', 'a+'))


def clip_subset_img_1k_base(args):
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='pretrained/open_clip_vit_h_14_laion2b_s32b_b79k.bin')
    model = model.visual
    model.cuda()
    model.eval()
    stepsize = 500

    class_list = open('datasets/place365/flist/class.txt').readlines()
    all_imgs=[]
    for retainname in class_list[:100]:
        tmp = glob.glob(os.path.join(args.data_root, 'val_256', retainname[1:].replace('\n', ''), '*.jpg'))
        all_imgs += tmp

    img_folder = './ckpt/original'
    os.makedirs(img_folder, exist_ok=True)
    for i in range(0, len(all_imgs), stepsize):
        print(i ,min(i+stepsize, len(all_imgs)))
        imgs_batch = all_imgs[i:min(i+stepsize, len(all_imgs))]
        latent_norm = run_clip_multi(imgs_batch, model, preprocess)
        for imgid, imgname in enumerate(imgs_batch):
            base_name = os.path.basename(imgname)
            np.savetxt(os.path.join(img_folder, base_name.replace('.jpg', '_clip_norm.txt')), latent_norm[imgid])


clip_subset_img_1k(args.img_folder, args)





