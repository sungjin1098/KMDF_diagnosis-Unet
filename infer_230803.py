import os
import copy
import glob
import torch
import argparse
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import skimage.io as skio
import cv2
import json
import warnings
import segmentation_models_pytorch as smp

from trs_model import *
from PIL import Image

warnings.filterwarnings("ignore")


def gen_index(input_size, patch_size, overlap_size):
    indices = []
    for k in range(2):
        z_range = list(range(0, input_size[k] - patch_size[k] + 1, overlap_size[k]))
        if input_size[k] - patch_size[k] > z_range[-1]:
            z_range.append(input_size[k] - patch_size[k])
        indices.append(z_range)
    return indices


def infer(opt):
    # Prepare for use of CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define model and loss function
    model = smp.Unet(encoder_name='mobilenet_v2', encoder_depth=5, encoder_weights=None,
                     decoder_use_batchnorm=True, decoder_channels=[256, 128, 64, 32, 16])

    # If you use MULTIPLE GPUs, use 'DataParallel'
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(opt.model_path))

    model = model.to(device)
    model.eval()

    Dec2 = Decoder(dim=64, n_upsample=2, n_residual=3, style_dim=8)
    Dec2.load_state_dict(torch.load(opt.dec_path))
    Dec2 = Dec2.to(device)
    Dec2.eval()
    
    Enc1 = Encoder(dim=64, n_downsample=2, n_residual=3, style_dim=8)
    Enc1.load_state_dict(torch.load(opt.enc_path))
    Enc1 = Enc1.to(device)
    Enc1.eval()
    
    img_path = opt.in_dir
    img_paths = sorted(glob.glob(f'{img_path}/*'))

    final_score = 0

    with torch.no_grad():
        for img_path in img_paths:
            img = torch.from_numpy(skio.imread(img_path)).permute(2, 0, 1).float().unsqueeze(0)
            img = img / 255

            pad = 256
            patch = 512

            # add margin
            img_pad = nn.functional.pad(img, (pad, pad, pad, pad), mode='reflect')

            # Overlapped patches are used for validation
            val_ind = gen_index(img_pad.size()[2:], [patch+pad+pad, patch+pad+pad], [patch, patch])

            h = img.size()[2]
            w = img.size()[3]
            x_remain = img.size()[2] % patch if img.size()[2] % patch != 0 else patch
            y_remain = img.size()[3] % patch if img.size()[3] % patch != 0 else patch

            imgs = torch.zeros(img.size())

            for xi in range(len(val_ind[0])):
                for yi in range(len(val_ind[1])):
                    x_small = img_pad[:, :, val_ind[0][xi]:val_ind[0][xi] + patch+pad+pad, val_ind[1][yi]:val_ind[1][yi] + patch+pad+pad].to(device)

                    x_out = model(x_small)
                    x_out_sg = torch.where(x_out > 0.5, 1/(1+torch.exp(-2.5*x_out)), 1/(1+torch.exp(-x_out)))

                    if xi != len(val_ind[0]) - 1 and yi == len(val_ind[1]) - 1:
                        imgs[:, :, xi * patch:(xi + 1) * patch, yi * patch:] = x_out_sg[:, :, pad:-pad, pad + patch - y_remain:-pad]
                    elif xi == len(val_ind[0]) - 1 and yi != len(val_ind[1]) - 1:
                        imgs[:, :, xi * patch:, yi * patch:(yi + 1) * patch] = x_out_sg[:, :, pad + patch - x_remain: -pad, pad:-pad]
                    elif xi == len(val_ind[0]) - 1 and yi == len(val_ind[1]) - 1:
                        imgs[:, :, xi * patch:, yi * patch:] = x_out_sg[:, :, pad + patch - x_remain: -pad, pad + patch - y_remain:-pad]
                    else:
                        imgs[:, :, xi * patch:(xi + 1) * patch, yi * patch:(yi + 1) * patch] = x_out_sg[:, :, pad:-pad, pad:-pad]

            # print and save score
            window = 256
            val_ind = gen_index(imgs.size()[2:], [window, window], [window-1, window-1])
            means = []
            for xi in range(len(val_ind[0])):
                for yi in range(len(val_ind[1])):
                    x_small = imgs[:, :, val_ind[0][xi]:val_ind[0][xi] + window, val_ind[1][yi]:val_ind[1][yi] + window]
                    means.append(x_small.mean().item())
            max_value = max(means)
            print(f'File: {img_path}, Score: {max_value}')

            if max_value > final_score:
                final_score = max_value

            ############
            # save cmap
            ############

            score_cmap = np.zeros((int(img.size()[2]), int(img.size()[3]), 4), np.uint8)

            # rgb to gray
            imgs = np.mean(np.array(imgs), 1)

            score_cmap[..., 0] = 0
            score_cmap[..., 1] = np.array(np.reshape(imgs, (1, img.size()[2], img.size()[3])).squeeze() * 255).astype(np.uint8)
            score_cmap[..., 2] = 0
            score_cmap[..., 3] = np.array(np.reshape(imgs, (1, img.size()[2], img.size()[3])).squeeze() * 255).astype(np.uint8)

            # save score map
            mappath = img_path.replace(opt.in_dir, opt.cmap_dir)
            cv2.imwrite(mappath[:-4] + '.png', score_cmap)

            ######################
            # Translate TPM to HE
            ######################
            pad = 128
            patch = 1024
            half = int(patch/2)

            h_remain = patch - h % patch if h % patch != 0 else 0
            w_remain = patch - w % patch if w % patch != 0 else 0

            imgs = []
            totensor_ = transforms.ToTensor()
            Tensor = torch.cuda.FloatTensor

            whole_img = F.pad(totensor_(Image.open(img_path)).unsqueeze(0), (0, w_remain, 0, h_remain), mode='reflect')
            whole_img = F.pad(whole_img, (pad, pad, pad, pad), mode='reflect')

            if h_remain == 0:
                num_x_centers = h // patch
            else:
                num_x_centers = h // patch + 1

            if w_remain == 0:
                num_y_centers = w // patch
            else:
                num_y_centers = w // patch + 1

            centers = []
            images = []

            with torch.no_grad():
                for i in range(1, num_x_centers + 1):
                    for j in range(1, num_y_centers + 1):
                        center = (pad + patch * i - half, pad + patch * j - half)
                        centers.append(center)
                        images.append(copy.deepcopy(whole_img[:, :, center[0] - half - pad: center[0] + half + pad, center[1] - half - pad: center[1] + half + pad]))

            style_2 = np.array(
                [[-0.50248137, 0.86058847, -0.82769679, -0.76093219, 0.62607271, 0.69616557, -0.15004669, -0.03822532]])
            style_2 = Variable(Tensor(style_2))

            # Generate samples
            for image in images:
                image = image.squeeze(0)

                with torch.no_grad():
                    c_code_1, _ = Enc1(Variable(image/255).type(Tensor).unsqueeze(0))
                    hedata = Dec2(c_code_1, style_2)
                imgs.append(hedata[:, :, pad:-pad, pad:-pad])

            v_imgs = []

            for i in range(num_x_centers):
                v_img = torch.cat(imgs[i * num_y_centers: (i + 1) * num_y_centers], 3)
                v_imgs.append(v_img)
            hedata = torch.cat(v_imgs, 2)

            if h_remain != 0:
                hedata = hedata[:, :, :-h_remain, :]
                if w_remain != 0:
                    hedata = hedata[:, :, :, :-w_remain]

            # to find a mask
            img_for_mask = cv2.imread(img_path)
            img_for_mask = cv2.cvtColor(img_for_mask, cv2.COLOR_BGR2GRAY)

            # get output image
            hepath = img_path.replace(opt.in_dir, opt.he_dir)
            hepath = hepath[:-3] + 'tif'
            trans = transforms.ToPILImage()
            hedata = np.array(trans(hedata.squeeze(0)))
            hedata = cv2.cvtColor(hedata, cv2.COLOR_RGB2BGR)

            # apply mask to the output image
            # (chosen) 1st version equation: 255-(x-0.01)**2, x<10
            # (reject) 2nd version equation: 255-(0.25x-0.01)**2 , x<30
            hedata = np.where(np.expand_dims(img_for_mask, axis=2) < 10, 255-(np.expand_dims(img_for_mask, axis=2)-0.01)**2, hedata)
            hedata = np.clip(np.round(hedata), 0, 255).astype(np.uint8)
            cv2.imwrite(hepath, hedata)

        js = {'Score': final_score}
        with open(opt.score_path, 'w') as of:
            json.dump(js, of)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, default='./Test_image', help="input-image directory")
    parser.add_argument("--he_dir", type=str, default='./Test_image_he', help="he-image save directory")
    parser.add_argument("--cmap_dir", type=str, default='./Test_image_cmap', help="contour version score-map save directory")
    parser.add_argument("--score_path", type=str, default='./score.json', help="score json save path")
    parser.add_argument("--model_path", type=str, default='./model_230803.pth', help="path of model parameters")
    parser.add_argument("--dec_path", type=str, default='./decoder.pth', help="path of decoder2 parameters")
    parser.add_argument("--enc_path", type=str, default='./encoder.pth', help="path of encoder1 parameters")
    opt = parser.parse_args()

    if not os.path.exists(opt.he_dir): os.makedirs(opt.he_dir)
    if not os.path.exists(opt.cmap_dir): os.makedirs(opt.cmap_dir)

    infer(opt)
