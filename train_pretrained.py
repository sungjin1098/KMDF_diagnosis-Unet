import os
import random
import math
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn
import torchvision
import skimage.io as skio
import matplotlib.pyplot as plt
import warnings

from torch import device
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

from util import gen_index, gen_index_random
from dataloader import KmdfDataset, KmdfDataset_valid
import segmentation_models_pytorch as smp

warnings.filterwarnings("ignore")

# Before running the code, modify '--data_dir' and '--save_name'
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="initial epoch")
parser.add_argument("--n_epoch", type=int, default=20001, help="last epoch")
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--lr", type=float, default=1e-4, help="initial epoch")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--ckt_period", type=int, default=50, help="period of saving model checkpoints w.r.t epochs")
parser.add_argument("--weight_decay", type=float, default=1e-6, help="initial epoch")
parser.add_argument("--data_dir", type=str, default="./dataset/230703_whole_split", help="dataset path")
parser.add_argument("--save_name", type=str, default='230705_02_batch32', help="save folder name")
opt = parser.parse_args()


# Save path: {saved_models, loss_curve, output_images}
root = f'./{opt.save_name}'
model_path = root + '/saved_models'
plot_path = root + '/loss_curve'
image_path = root + '/output_images'
os.makedirs(model_path, exist_ok=True)
os.makedirs(plot_path, exist_ok=True)
os.makedirs(image_path, exist_ok=True)

# Prepare for use of CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define model and loss function
model = smp.Unet(encoder_name='mobilenet_v2', encoder_depth=5, encoder_weights='imagenet',
                 decoder_use_batchnorm=True, decoder_channels=[256, 128, 64, 32, 16])
# from torchsummary import summary
# summary(model.to(device), (3, 512, 512))
# print(model)
# for name, p in model.named_parameters():
#     print(name)
# # 18,438,545

# If you use MULTIPLE GPUs, use 'DataParallel'
model = nn.DataParallel(model)

# .to(device)
model = model.to(device)
criterion = nn.BCEWithLogitsLoss().to(device)

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Load pretrained model
model.load_state_dict(torch.load("./model.pth"))

# Define data transform and loader
train_data = KmdfDataset(f"{opt.data_dir}/train")
train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)
valid_data = KmdfDataset_valid(f"{opt.data_dir}/test")
valid_loader = DataLoader(dataset=valid_data, batch_size=1, shuffle=False, num_workers=opt.n_cpu)

# is_best
best_val_acc = 0
epochs_since_improvement = 0

# # Overlapped patches are the training patches
# ind = gen_index(input_size=[1024, 1024], patch_size=[512, 512], overlap_size=[256, 256])

# To draw loss graph
train_loss_ep = []
train_acc_ep = []
val_loss_ep = []
val_acc_ep = []
epoch_ep = []

# x, y = next(iter(train_loader))
# x_one = x[:, :, :512, :512]
# y_one = y[:, :, :512, :512]
# for i in range(x.size(0)):
#     skio.imsave(f"{image_path}/x_small_{i}_train.tif", x_one[i].permute(1, 2, 0).cpu().numpy())
#     skio.imsave(f"{image_path}/y_small_{i}_train.tif", y_one[i, 0].detach().cpu().numpy())


def random_rotate_8(a, b, half_receptive):
    degree = random.uniform(-8, 8)
    if degree < 0:
        a = torchvision.transforms.functional.rotate(a, angle=360+degree,
                                                                 interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
        b = torch.nn.functional.pad(input=b, pad=(half_receptive, half_receptive, half_receptive, half_receptive), mode='reflect')
        b = torchvision.transforms.functional.rotate(b, angle=360+degree,
                                                                 interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        # crop size
        crop = math.ceil(a.size()[2] * np.sin(math.radians((-degree))))
        a = a[:, :, crop:-crop, crop:-crop]
        b = b[:, :, crop:-crop, crop:-crop]
        # skio.imsave("./352save.png", a[0].permute(1, 2, 0).detach().cpu().numpy())
    elif degree > 0:
        a = torchvision.transforms.functional.rotate(a, angle=degree,
                                                                 interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
        b = torch.nn.functional.pad(input=b, pad=(half_receptive, half_receptive, half_receptive, half_receptive), mode='reflect')
        b = torchvision.transforms.functional.rotate(b, angle=degree,
                                                                 interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        # crop size
        crop = math.ceil(a.size()[2] * np.sin(math.radians(degree)))
        a = a[:, :, crop:-crop, crop:-crop]

        b = b[:, :, crop:-crop, crop:-crop]
        # skio.imsave("./8save.png", a[0].permute(1, 2, 0).detach().cpu().numpy())
    else:
        crop = 0
        pass

    return a, b, crop


for epoch in range(opt.epoch, opt.n_epoch):
    epoch_ep.append(epoch+1)
    ### training
    # is_best

    model.train()
    # total_loss, total_cnt, correct_cnt = np.array([0.0, 0.0, 0.0]), 0.0, 0.0
    # total_label_cnt = 0.0

    total_loss = 0.0
    total_acc = 0.0
    total_cnt = 0
    for i, (x, y) in enumerate(tqdm(train_loader)):
        # x.size() = [batch_size, 4, 1024, 1024]
        # by considering the receptive field, mirror padding (256, 256) ==> [batch_size, 4, 1536, 1536]
        half_receptive = 64
        x = torch.nn.functional.pad(input=x, pad=(half_receptive, half_receptive, half_receptive, half_receptive), mode='reflect')

        # random rotate
        x, y, crop = random_rotate_8(x, y, half_receptive)


        # print(x.size()): [batch_size, 3, 1536, 1536]
        ind = gen_index_random(x.size()[2:], 512)
        # print(ind): random index (x,y) 4개

        for xi in range(len(ind[0])):
            x_small = x[:, :, ind[0][xi]:ind[0][xi]+512, ind[1][xi]:ind[1][xi]+512].to(device)
            y_small = y[:, :, ind[0][xi]:ind[0][xi]+512, ind[1][xi]:ind[1][xi]+512].to(device)
            # print(x_small.size(), y_small.size()): 4, 3, 1024, 1024 and 4, 1, 512, 512

            # Zero gradients
            optimizer.zero_grad()

            # model output
            x_out = model(x_small)
            # have to use only center region
            # x_out = x_out[:, :, 128:-128, 128:-128]

            loss = criterion(x_out, y_small)

            with torch.no_grad():
                x_final = (torch.nn.functional.sigmoid(x_out) > 0.5).float()
                acc = (x_final == y_small).float().mean()

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += acc.item()
            total_cnt += 1

    train_loss_ep.append(total_loss / total_cnt)
    train_acc_ep.append(total_acc / total_cnt)

    print(f"[{epoch}/{opt.n_epoch}] Train Loss: {train_loss_ep[epoch]}")

    # Save checkpoint and images
    if epoch % opt.ckt_period == 0:
        torch.save(model.state_dict(), model_path + "/model_%d.pth" % epoch)

        # Segmentation map result = thresholding the output
        x_out_sg = torch.nn.functional.sigmoid(x_out)
        x_final = (torch.nn.functional.sigmoid(x_out) > 0.5).float()

        for i in range(x_out.size(0)):
            # images = [x_out[i, 0], x_out_sg[i, 0], x_final[i, 0]]
            # images = torch.cat(images, dim=0)
            # images = images.detach().cpu().numpy().astype('uint8')
            # skio.imsave(f"{image_path}/{epoch}_{i}_train.tif", images)
            skio.imsave(f"{image_path}/{epoch}_{i}_x_small_train.tif", x_small[i].permute(1,2,0).detach().cpu().numpy())
            skio.imsave(f"{image_path}/{epoch}_{i}_x_out_sg_train.tif", x_out_sg[i, 0].detach().cpu().numpy())
            skio.imsave(f"{image_path}/{epoch}_{i}_x_out_train.tif", x_out[i, 0].detach().cpu().numpy())
            skio.imsave(f"{image_path}/{epoch}_{i}_x_final_train.tif", x_final[i, 0].detach().cpu().numpy())
            skio.imsave(f"{image_path}/{epoch}_{i}_y_small_train.tif", y_small[0, 0].detach().cpu().numpy())

        # torchvision.utils.save_image(x_small[0].permute(1, 2, 0), f"{image_path}/{epoch}_x_small_train.jpg")

    ### validation
    model.eval()

    with torch.no_grad():
        val_total_loss = 0.0
        val_total_acc = 0.0
        val_cnt = 0
        for i, (x, y) in enumerate(tqdm(valid_loader)):
            # x = torch.nn.functional.pad(input=x, pad=(128, 128, 128, 128), mode='reflect')

            # Overlapped patches are used for validation
            val_ind = gen_index(x.size()[2:], [512, 512], [512, 512])

            for xi in range(len(val_ind[0])):
                for yi in range(len(val_ind[1])):
                    x_small = x[:, :, val_ind[0][xi]:val_ind[0][xi] + 512, val_ind[1][yi]:val_ind[1][yi] + 512].to(device)
                    y_small = y[:, :, val_ind[0][xi]:val_ind[0][xi] + 512, val_ind[1][yi]:val_ind[1][yi] + 512].to(device)
                    # print(x_small.size(), x_small_save.size(), y_small.size())

                    x_out = model(x_small)
                    x_out_sg = torch.nn.functional.sigmoid(x_out)
                    # x_out = x_out[:, :, 128:-128, 128:-128]

                    # Segmentation map result = thresholding the output
                    x_final = (torch.nn.functional.sigmoid(x_out) > 0.5).float()

                    val_loss = criterion(x_out, y_small)

                    val_total_loss += val_loss.item()
                    val_cnt += 1

                    with torch.no_grad():
                        acc = (x_final == y_small).float().mean()

                    val_total_acc += acc.item()

                    # skio.imsave(f"./x_small/{test_cnt}_x_small.tif", x_small[0].permute(1, 2, 0).cpu().numpy())
                    # skio.imsave(f"./x_out/{test_cnt}_x_out.tif", x_out[0, 0].cpu().numpy())
                    # skio.imsave(f"./x_final/{test_cnt}_x_final.tif", x_final[0, 0].cpu().numpy())
                    # skio.imsave(f"./y_small/{test_cnt}_y_small.tif", y_small[0, 0].cpu().numpy())
                    if epoch % opt.ckt_period == 0:
                        if xi % 4 == 0 and yi % 4 == 0:
                            skio.imsave(f"{image_path}/{epoch}_{xi}_{yi}_x_small_val.tif", x_small[0].permute(1, 2, 0).cpu().numpy())
                            skio.imsave(f"{image_path}/{epoch}_{xi}_{yi}_x_out_val.tif", x_out[0, 0].cpu().numpy())
                            skio.imsave(f"{image_path}/{epoch}_{xi}_{yi}_x_out_sg_val.tif", x_out_sg[0, 0].cpu().numpy())
                            skio.imsave(f"{image_path}/{epoch}_{xi}_{yi}_x_final_val.tif", x_final[0, 0].cpu().numpy())
                            skio.imsave(f"{image_path}/{epoch}_{xi}_{yi}_y_small_val.tif", y_small[0, 0].cpu().numpy())

        val_loss_ep.append(val_total_loss / val_cnt)
        val_acc_ep.append(val_total_acc / val_cnt)

        # is_best
        is_best = val_acc_ep[epoch] > best_val_acc
        best_val_acc = max(val_acc_ep[epoch], best_val_acc)

    if not is_best:
        epochs_since_improvement += 1
        print("\nEpochs since last improvement: %d\n" % epochs_since_improvement)
    else:
        epochs_since_improvement = 0
    if is_best:
        torch.save(model.state_dict(), model_path + f"/best_model_{epoch}_{best_val_acc:.4f}.pth")

    if epoch % 1 == 0:
        print(f"[{epoch}/{opt.n_epoch}] Val Loss : {val_loss_ep[epoch]}, Best val acc: {best_val_acc} \n")


    # Save images
    if epoch % 1 == 0:
        images = []

        plt.clf()
        epoch_ep_n = np.array(epoch_ep)
        train_loss_ep_n = np.array(train_loss_ep)
        val_loss_ep_n = np.array(val_loss_ep)
        plt.plot(epoch_ep_n, train_loss_ep_n, lw=0.75, color='red', label='train_loss')
        plt.plot(epoch_ep_n, val_loss_ep_n, lw=0.75, color='blue', label='valid_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper left', prop={'size': 6})
        plt.savefig(plot_path + '/loss_curve.png')

        plt.clf()
        epoch_ep_n = np.array(epoch_ep)
        train_acc_ep_n = np.array(train_acc_ep)
        val_acc_ep_n = np.array(val_acc_ep)
        plt.plot(epoch_ep_n, train_acc_ep_n, lw=0.75, color='red', label='train_acc')
        plt.plot(epoch_ep_n, val_acc_ep_n, lw=0.75, color='blue', label='valid_acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='upper left', prop={'size': 6})
        plt.savefig(plot_path + '/accuracy_curve.png')
