We modified this repo to reproduce the experiments for the paper <a href="https://openreview.net/pdf?id=9hjVoPWPnh">Machine Unlearning for Image-to-Image Generative Models</a> (ICLR 2024).


You shall run the code in the current dir: `i2i_Palette-Image-to-Image-Diffusion-Models/`

# Setup


### Download Pre-trained Model

| Dataset   | Task       | URL                                                          |
| --------- | ---------- | ------------------------------------------------------------ |
| Places2   | Inpainting | [Google Drive](https://drive.google.com/file/d/1lxkcyMc6-VUDvNnOg3krryLoGMpBChuM/view) |



### Configurations
- All the configs used in our experiments are under `config/` folder. We will mention them later for their specific usage.

- You need to modify the `data_root` to the above `DATA_DIR` correspondingly.

# Unlearn
### Unlearn with real $D_R$
The configs are within `config/inpainting_places2_unlearn.json`.

`python unlearn.py -p train -c config/inpainting_places2_unlearn.json --forget_alpha=X --fix_decoder=X --learn_noise X --learn_others X --ckpt CKPT_PATH`

- ckpt: the location of pretrained ckpt
- forget_alpha: the alpha for the forget loss
- fix_decoder: if set `1`, we will only tune the encoder
- max_loss: if set `1`, we will maximize the loss on the forget set; this is one of baseline
- learn_noise: if set `1`, we will assign random noise as the label for forget set; this is one of baseline
- learn_others: if set `1`, we will assign retain images as the label for forget set; this is one of baseline

|Approach|--fix_decoder|--max_loss|--learn_noise|--learn_others|
|-|-|-|-|-|
|  ***Ours***  | 1   |  0  |  0  |  0  |
|  Retain Label |  0  |  0  |  0  |  1  |
|  Random label  |  0  |  0  |  1  |  0  |
|  Max loss  |  0  |  1  |  0  |  0  |
|  Random Ecnoder  |  1  |  0  |  1  |  0  |



**Example [Our apporach]**:

`python unlearn.py -p train -c config/inpainting_places2_unlearn.json --forget_alpha 0.1 --fix_decoder=1 --learn_noise 0 --learn_others 0 --ckpt 16_Network.pth`

### Unlearn with Proxy $D_R$
The input arguments are the same as above but with different configs `config/inpainting_places2_unlearn_open.json`.

**Example [Our apporach]**:

`python unlearn.py -p train -c config/inpainting_places2_unlearn_open.json --forget_alpha 0.1 --fix_decoder=1 --learn_noise 0 --learn_others 0 --ckpt 16_Network.pth`

# Test
### **Generate images**

The configs are within `config/inpainting_places2_unlearn_test.json`.


The following script will generate 10K images from the test set of Places-365

`python run.py -p test -c config/inpainting_places2_unlearn_test.json --ckpt $CKPT_PATH`

- Example: test the model before unlearning:

	- `python run.py -p test -c config/inpainting_places2_unlearn_test.json --ckpt 16_Network.pth`

### **Compute FID and IS**

As there is no inceptionV3 model available for Places2, we use the ResNet50 to compute FID instead. 
Download the model from Places2 [Github](https://github.com/CSAILVision/places365), [Direct Link](http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar), and locate into the `pretrained` folder.

Then run the following code:

`python eval365.py --dst $PATH_OF_GENERATED_IMG_FOLDERS`

`$PATH_OF_GENERATED_IMG_FOLDERS` is the path of the tested checkpoint by removing the post-fix '.pth'

- Example: test the model before unlearning:

	- `python eval365.py --dst 16_Network/`
- Results: The results for FID and IS score will be stored at `fid_is_eval.csv`.

### **Computer CLIP score**

Download the pretrained CLIP-model:
```
mkdir pretrained
wget https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/open_clip_pytorch_model.bin
mv open_clip_pytorch_model.bin pretrained/open_clip_vit_h_14_laion2b_s32b_b79k.bin
```

Run the generated images:

`python clip_embed.py --img_folder $PATH_OF_GENERATED_IMG_FOLDERS`


- Example: test the model before unlearning:

	- `python clip_embed.py --img_folder 16_Network/`
- Results: The results for CLIP score will be stored at `clip_cosine.csv`.


### **T-SNE**

After CLIP, run T-SNE with CLIP embedding

`python tsne.py --ckpt_folder $PATH_OF_GENERATED_IMG_FOLDERS`

- Example: run the T-SNE analysis for the model before unlearning:

	- `python tsne.py --ckpt_folder 16_Network/`



