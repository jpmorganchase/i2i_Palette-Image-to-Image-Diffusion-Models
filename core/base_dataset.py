import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(folder):
    if isinstance(folder, list):
        return folder
    elif os.path.isfile(folder):
        images = [i for i in np.genfromtxt(folder, dtype=str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(folder), '%s is not a valid directory' % folder
        for root, _, fnames in sorted(os.walk(folder)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images

def pil_loader(path):
    return Image.open(path).convert('RGB')

class BaseDataset(data.Dataset):
    def __init__(self, data_root, image_size=[256, 256], loader=pil_loader, tfs=None):
        self.imgs = make_dataset(data_root)
        if tfs is None:
            self.tfs = transforms.Compose([
                    transforms.Resize((image_size[0], image_size[1])),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        else:
            self.tfs = tfs
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        return img

    def __len__(self):
        return len(self.imgs)
