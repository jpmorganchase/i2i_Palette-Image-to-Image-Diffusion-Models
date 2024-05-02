import os
import glob
import argparse
from cleanfid import fid
from core.base_dataset import BaseDataset
from models.metric import inception_score, inception_score_place365
from torchvision import transforms as trn
import random


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dst', type=str, help='Generated images directory')
    parser.add_argument('--mask_ratio', default=0.5, type=float, help='masking ratio w.r.t. one dimension')
    args = parser.parse_args()

    tfs = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    class_list = open('datasets/place365/flist/class.txt').read().splitlines()
    retain_class = class_list[:50]
    forget_class = class_list[50:100]
    
    dst=[]
    for retainname in retain_class:
        tmp = glob.glob(os.path.join(args.dst, f'results/test/0/{args.mask_ratio:.2f}/Out'+retainname.replace('/', '_')+'_Places365*.jpg'))
        dst += tmp

    random.shuffle(dst)
    src = [img.replace('Out_', 'GT_') for img in dst]
    fid_score = fid.compute_fid(src, dst)
    try:
        is_mean, is_std = inception_score_place365(BaseDataset(dst, tfs=tfs), cuda=True, batch_size=8, resize=False, splits=10)
    except:
        is_mean, is_std = inception_score(BaseDataset(dst, tfs=tfs), cuda=True, batch_size=8, resize=False, splits=10)
    metric=[args.dst, fid_score, is_mean, is_std]
    print('FID: {}'.format(fid_score))
    print('IS:{} {}'.format(is_mean, is_std))


    dst=[]
    for forgetname in forget_class:
        tmp = glob.glob(os.path.join(args.dst, f'results/test/0/{args.mask_ratio:.2f}/Out'+forgetname.replace('/', '_')+'_Places365*.jpg'))
        dst += tmp
    random.shuffle(dst)
    src = [img.replace('Out_', 'GT_') for img in dst]
    fid_score = fid.compute_fid(src, dst)
    try:
        is_mean, is_std = inception_score_place365(BaseDataset(dst, tfs=tfs), cuda=True, batch_size=8, resize=False, splits=10)
    except:    
        is_mean, is_std = inception_score(BaseDataset(dst, tfs=tfs), cuda=True, batch_size=8, resize=False, splits=10)
    print('FID: {}'.format(fid_score))
    print('IS:{} {}'.format(is_mean, is_std))
    metric += [fid_score, is_mean, is_std]
    print('\t'.join(str(a) for a in metric), file=open('fid_is_eval.csv', 'a+'))
    
     



