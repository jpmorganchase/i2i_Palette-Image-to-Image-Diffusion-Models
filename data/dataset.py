import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np
import glob
import random
import pandas as pd
from .util.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(imgdir):
    if os.path.isfile(imgdir):
        images = open(imgdir, 'r').read().splitlines()
    else:
        images = []
        assert os.path.isdir(imgdir), '%s is not a valid directory' % imgdir
        for root, _, fnames in sorted(os.walk(imgdir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images

def pil_loader(path):
    return Image.open(path).convert('RGB')

class InpaintDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class UncroppingDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'manual':
            mask = bbox2mask(self.image_size, self.mask_config['shape'])
        elif self.mask_mode == 'fourdirection' or self.mask_mode == 'onedirection':
            mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode=self.mask_mode))
        elif self.mask_mode == 'hybrid':
            if np.random.randint(0,2)<1:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='onedirection'))
            else:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='fourdirection'))
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class ColorizationDataset(data.Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, image_size=[224, 224], loader=pil_loader):
        self.data_root = data_root
        flist = make_dataset(data_flist)
        if data_len > 0:
            self.imgs = flist[:int(data_len)]
        else:
            self.imgs = flist
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        file_name = str(self.imgs[index]).zfill(5) + '.png'

        img = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'color', file_name)))
        cond_image = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'gray', file_name)))

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['path'] = file_name
        return ret

    def __len__(self):
        return len(self.imgs)


class ForgetInpaintDataset(data.Dataset):
    def __init__(self, data_root=None, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader, number_img_per_class=5000):


        class_list = open('datasets/place365/flist/class.txt').read().splitlines()
        retain_class = class_list[:50]
        forget_class = class_list[50:100]
        retain_imgs=[]
        forget_imgs=[]
        for retainname in retain_class:
            tmp = glob.glob(os.path.join(data_root, 'train_256', retainname[1:], '*.jpg'))
            retain_imgs += tmp[:number_img_per_class]
        for forgetname in forget_class:
            tmp = glob.glob(os.path.join(data_root, 'train_256', forgetname[1:], '*.jpg'))
            forget_imgs += tmp[:number_img_per_class]
        
        retain_imgs = retain_imgs[:len(forget_imgs)]
        forget_imgs = forget_imgs
        self.imgs = retain_imgs + forget_imgs
        self.labels = [1.0]*len(retain_imgs)+[-1.0]*len(forget_imgs)

        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])

        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret, self.labels[index]

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class TestForgetInpaintDataset(data.Dataset):
    def __init__(self, data_root=None, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader, number_img_per_class=100):


        class_list = open('datasets/place365/flist/class.txt').read().splitlines()
        retain_imgs=[]
        forget_imgs=[]
        for retainname in class_list[:50]:
            tmp = glob.glob(os.path.join(data_root, 'val_256', retainname[1:], '*.jpg'))
            retain_imgs += tmp[:number_img_per_class]
        for forgetname in class_list[50:100]:
            tmp = glob.glob(os.path.join(data_root, 'val_256', forgetname[1:], '*.jpg'))
            forget_imgs += tmp[:number_img_per_class]

        self.imgs = forget_imgs+retain_imgs
        self.labels = [-1.0]*len(forget_imgs)+[1.0]*len(retain_imgs)

        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.mask_ratio = self.mask_config['mask_ratio']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret, self.labels[index]

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            crop_h, crop_w = int(h * self.mask_ratio), int(w * self.mask_ratio)
            mask = bbox2mask(self.image_size, ((h - crop_h) // 2, (w - crop_w) // 2, crop_h, crop_w))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class OpenForgetInpaintDataset(data.Dataset):
    def __init__(self, data_root=None, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader, number_img_per_class=200):
        class_list = open('datasets/place365/flist/class.txt').read().splitlines()
        retain_class = class_list[100:]
        forget_class = class_list[50:100]
        retain_imgs=[]
        forget_imgs=[]
        for retainname in retain_class:
            tmp = glob.glob(os.path.join(data_root, 'train_256', retainname[1:], '*.jpg'))
            retain_imgs += tmp[:number_img_per_class]
        for forgetname in forget_class:
            tmp = glob.glob(os.path.join(data_root, 'train_256', forgetname[1:], '*.jpg'))
            forget_imgs += tmp[:number_img_per_class]
        self.imgs = retain_imgs + forget_imgs
        self.labels = [1.0]*len(retain_imgs)+[-1.0]*len(forget_imgs)


        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret, self.labels[index]

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class ForgetUncroppingDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader, number_img_per_class=100):
        data_root=None
        if data_root is not None:
            imgs = make_dataset(data_root)
            if data_len > 0:
                self.imgs = imgs[:int(data_len)]
            else:
                self.imgs = imgs
            self.labels = [1.0]*len(imgs) 
        else:
            csv_path = 'datasets/place365/flist/ForgetUncroppingDataset_{}_{}_{}.csv'.format(number_img_per_class, data_len, image_size[0])
            if os.path.isfile(csv_path):
                data_info = pd.read_csv(csv_path, sep=',', header=0, names = ['imgs', 'labels'])
                self.imgs = list(data_info.imgs)
                self.labels = list(data_info.labels)
            else:
                class_list = open('datasets/place365/flist/class.txt').read().splitlines()
                retain_class = class_list[:50]
                forget_class = class_list[50:100]
                retain_imgs=[]
                forget_imgs=[]
                for retainname in retain_class:
                    tmp = glob.glob(os.path.join(data_root, 'train_256', retainname[1:], '*.jpg'))
                    random.shuffle(tmp)
                    retain_imgs += tmp[:number_img_per_class]
                for forgetname in forget_class:
                    tmp = glob.glob(os.path.join(data_root, 'train_256', forgetname[1:], '*.jpg'))
                    random.shuffle(tmp)
                    forget_imgs += tmp[:number_img_per_class]
                self.imgs = retain_imgs + forget_imgs
                self.labels = [1.0]*len(retain_imgs)+[-1.0]*len(forget_imgs)
                datainfo = pd.DataFrame(list(zip(self.imgs, self.labels)), columns =['imgs', 'labels'])
                datainfo.to_csv(csv_path, index=False,)

        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret, self.labels[index]

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'manual':
            mask = bbox2mask(self.image_size, self.mask_config['shape'])
        elif self.mask_mode == 'fourdirection' or self.mask_mode == 'onedirection':
            mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode=self.mask_mode))
        elif self.mask_mode == 'hybrid':
            if np.random.randint(0,2)<1:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='onedirection'))
            else:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='fourdirection'))
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class ForgetColorizationDataset(data.Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, image_size=[224, 224], loader=pil_loader, number_img_per_class=100):
        data_root=None
        if data_root is not None:
            imgs = make_dataset(data_root)
            if data_len > 0:
                self.imgs = imgs[:int(data_len)]
            else:
                self.imgs = imgs
            self.labels = [1.0]*len(imgs) 
        else:
            csv_path = 'datasets/place365/flist/ForgetColorizationDataset_{}_{}_{}.csv'.format(number_img_per_class, data_len, image_size[0])
            if os.path.isfile(csv_path):
                data_info = pd.read_csv(csv_path, sep=',', header=0, names = ['imgs', 'labels'])
                self.imgs = list(data_info.imgs)
                self.labels = list(data_info.labels)
            else:
                class_list = open('datasets/place365/flist/class.txt').read().splitlines()
                retain_class = class_list[:50]
                forget_class = class_list[50:100]
                retain_imgs=[]
                forget_imgs=[]
                for retainname in retain_class:
                    tmp = glob.glob(os.path.join(data_root, 'train_256', retainname[1:], '*.jpg'))
                    random.shuffle(tmp)
                    retain_imgs += tmp[:number_img_per_class]
                for forgetname in forget_class:
                    tmp = glob.glob(os.path.join(data_root, 'train_256', forgetname[1:], '*.jpg'))
                    random.shuffle(tmp)
                    forget_imgs += tmp[:number_img_per_class]
                self.imgs = retain_imgs + forget_imgs
                self.labels = [1.0]*len(retain_imgs)+[-1.0]*len(forget_imgs)
                datainfo = pd.DataFrame(list(zip(self.imgs, self.labels)), columns =['imgs', 'labels'])
                datainfo.to_csv(csv_path, index=False,)

        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        file_name = str(self.imgs[index]).zfill(5) + '.png'

        img = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'color', file_name)))
        cond_image = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'gray', file_name)))

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['path'] = file_name
        return ret, self.labels[index]

    def __len__(self):
        return len(self.imgs)


