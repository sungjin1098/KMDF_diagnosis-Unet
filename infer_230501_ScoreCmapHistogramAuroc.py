import os
import copy
import glob
import torch
import argparse
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import json
import warnings
import segmentation_models_pytorch as smp

from trs_model import *
from PIL import Image
from skimage import io
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


def get_tpr(y_true, y_scores, threshold):
    predict_positive_num = len(y_scores[y_scores >= threshold])
    tp = len([x for x in y_true[:predict_positive_num] if x == 1])
    ground_truth = len(y_true[y_true == 1])
    try:
        tpr = tp / ground_truth
    except ZeroDivisionError:
        tpr = 0
    return tpr


def get_fpr(y_true, y_scores, threshold):
    predict_positive_num = len(y_scores[y_scores >= threshold])
    fp = len([x for x in y_true[:predict_positive_num] if x == 0])
    ground_negative = len(y_true[y_true == 0])
    try:
        fpr = fp / ground_negative
    except ZeroDivisionError:
        fpr = 0
    return fpr


def roc_plot(y_true, y_scores):
    tpr, fpr = [], []

    for _ in y_scores:  # y_scores 를 thresholds 처럼 사용했음
        tpr.append(get_tpr(y_true, y_scores, _))
        fpr.append(get_fpr(y_true, y_scores, _))

    plt.clf()
    plt.plot(fpr, tpr, color='r', linewidth=3.0, label='ROC')
    plt.scatter(fpr, tpr, color='r')
    plt.plot([0, 1], [0, 1], color='k', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    AUROC = roc_auc_score(y_true, y_scores)
    plt.title(f'ROC Curve (AUROC: {AUROC:.3f})')
    plt.savefig(opt.roc_path)


def gen_index(input_size, patch_size, overlap_size):
    indices = []
    for k in range(2):
        z_range = list(range(0, input_size[k] - patch_size[k] + 1, overlap_size[k]))
        if input_size[k] - patch_size[k] > z_range[-1]:
            z_range.append(input_size[k] - patch_size[k])
        indices.append(z_range)
    return indices


def infer(opt):
    # For confution matrix
    cnt = 0
    TP_cnt = 0
    FN_cnt = 0
    FP_cnt = 0
    TN_cnt = 0

    # For histogram
    N = []
    T = []

    # To save score information
    dictionary = dict()

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
            # For confusion matrix
            N_or_T_GT = img_path.split('/')[-1][5]
            img_name = img_path.split('/')[-1]

            img = torch.from_numpy(io.imread(img_path)).permute(2, 0, 1).float().unsqueeze(0)
            img = img / 255

            pad = 256
            patch = 512

            # add margin
            img_pad = nn.functional.pad(img, (pad, pad, pad, pad), mode='reflect')

            # Overlapped patches are used for validation
            val_ind = gen_index(img_pad.size()[2:], [patch+pad+pad, patch+pad+pad], [patch, patch])

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

            # To save score of each image
            dictionary[img_path.split('/')[-1]] = max_value

            # For confusion matrix
            if N_or_T_GT == 'N':
                N.append(max_value)
            elif N_or_T_GT == 'T':
                T.append(max_value)
            else:
                print("ERROR: Label cannot be determined!")
                print(img.split('/')[-1][:-4])
                exit()

            # Threshold to determine if the image is N or T
            mean_threshold = 0.5

            # For confusion matrix
            if max_value >= mean_threshold and N_or_T_GT == 'N':
                FP_cnt += 1
                save_img = io.imread(img_path)
                io.imsave(f'{opt.misclass_dir}/FP/{img_name}', save_img)
            elif max_value >= mean_threshold and N_or_T_GT == 'T':
                TP_cnt += 1
            elif max_value < mean_threshold and N_or_T_GT == 'N':
                TN_cnt += 1
            elif max_value < mean_threshold and N_or_T_GT == 'T':
                FN_cnt += 1
                save_img = io.imread(img_path)
                io.imsave(f'{opt.misclass_dir}/FN/{img_name}', save_img)
            else:
                print('ERROR: image-level score cannot be classified!')
                print(img.split('/')[-1][:-4])
                exit()

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

        # Save score.json
        with open(opt.score_path, 'w') as of:
            json.dump(dictionary, of, indent="\t")

        # Save ConfusionMatrix.json
        with open(opt.matrix_path, 'w') as of:
            try:
                Sensitivity = TP_cnt / (TP_cnt + FN_cnt)
            except ZeroDivisionError:
                Sensitivity = 0
            try:
                Specificity = TN_cnt / (TN_cnt + FP_cnt)
            except ZeroDivisionError:
                Specificity = 0
            try:
                Precision = TP_cnt / (TP_cnt + FP_cnt)
            except ZeroDivisionError:
                Precision = 0
            try:
                Negative_Predictive_Value = TN_cnt / (TN_cnt + FN_cnt)
            except ZeroDivisionError:
                Negative_Predictive_Value = 0

            dictionary2 = {
                'TP': TP_cnt,
                'TN': TN_cnt,
                'FP': FP_cnt,
                'FN': FN_cnt,
                'Sensitivity': Sensitivity,
                'Specificity': Specificity,
                'Precision': Precision,
                'Negative Predictive Value': Negative_Predictive_Value,
                'Accuracy': (TP_cnt + TN_cnt) / (TP_cnt + TN_cnt + FP_cnt + FN_cnt)
            }
            json.dump(dictionary2, of, indent="\t")

        # Save Histogram.png
        plt.clf()
        plt.hist(N, alpha=0.5, label='Normal', range=(0, 1), bins=20)
        plt.hist(T, alpha=0.5, label='Tumor', range=(0, 1), bins=20)
        plt.xlabel('Probability of being classified as Tumor (0~1)')
        plt.ylabel('# of images')
        plt.legend()
        plt.savefig(opt.his_path)

        # Save ROC.png
        y_scores = N + T
        y_true = [0]*len(N) + [1]*len(T)
        y_scores_sorted = np.sort(y_scores)[::-1]
        y_scores_index = np.argsort(y_scores)[::-1]
        y_true_sorted = [y_true[i] for i in y_scores_index]
        roc_plot(np.array(y_true_sorted), np.array(y_scores_sorted))


if __name__ == '__main__':
    ### INPUT is train? valid? test? ###
    DATA = 'train'

    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, default=f'./230517_{DATA}/image', help="input-image directory")
    parser.add_argument("--cmap_dir", type=str, default=f'./230517_{DATA}_cmap', help="contour version score-map save directory")
    parser.add_argument("--score_path", type=str, default=f'./230517_{DATA}_score.json', help="score json save path")
    parser.add_argument("--matrix_path", type=str, default=f'./230517_{DATA}_ConfusionMatrix.json', help="information to draw confusion matrix")
    parser.add_argument("--roc_path", type=str, default=f'./230517_{DATA}_ROC.png', help="ROC curve and AUROC")
    parser.add_argument("--his_path", type=str, default=f'./230517_{DATA}_Histogram.png', help="histogram of classifiaction")
    parser.add_argument("--misclass_dir", type=str, default=f'./230517_{DATA}_misclassified', help="save path of FP and FN images")

    parser.add_argument("--model_path", type=str, default='./model_230203.pth', help="path of model parameters")
    parser.add_argument("--dec_path", type=str, default='./decoder.pth', help="path of decoder2 parameters")
    parser.add_argument("--enc_path", type=str, default='./encoder.pth', help="path of encoder1 parameters")
    opt = parser.parse_args()

    if not os.path.exists(opt.cmap_dir): os.makedirs(opt.cmap_dir)
    if not os.path.exists(opt.misclass_dir): os.makedirs(opt.misclass_dir)
    if not os.path.exists(f'{opt.misclass_dir}/FP'): os.makedirs(f'{opt.misclass_dir}/FP')
    if not os.path.exists(f'{opt.misclass_dir}/FN'): os.makedirs(f'{opt.misclass_dir}/FN')

    infer(opt)
