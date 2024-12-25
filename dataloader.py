import torch
import random
import numpy as np
import skimage.io as skio
import glob
import cv2
# from torchvision.utils import save_image


class KmdfDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.path = path
        self.img_list = glob.glob(f"{path}/image/*")
        # print(self.img_list)

    # random horizontal flip
    @staticmethod
    def random_horizontal_flip(a, b):
        # probability 0.5
        if random.random() < 0.5:
            # stride를 -1로 해서 역정렬
            a = a[:, ::-1].copy()
            b = b[:, ::-1].copy()
        return a, b

    # random vertical flip
    @staticmethod
    def random_vertical_flip(a, b):
        if random.random() < 0.5:
            a = a[::-1, :].copy()
            b = b[::-1, :].copy()
        return a, b

    # random rotate 90 degrees
    @staticmethod
    def random_rotate(x, rot_times):
        x = np.rot90(x, rot_times, axes=(1,0)).copy()
        return x

    @staticmethod
    def random_rotate_90(a, b):
        if random.random() < 0.25:
            a = KmdfDataset.random_rotate(a, 1)
            b = KmdfDataset.random_rotate(b, 1)
        elif random.random() < 0.5:
            a = KmdfDataset.random_rotate(a, 2)
            b = KmdfDataset.random_rotate(b, 2)
        elif random.random() < 0.75:
            a = KmdfDataset.random_rotate(a, 3)
            b = KmdfDataset.random_rotate(b, 3)
        return a, b

    @staticmethod
    def random_gaussian(a):
        std = random.uniform(0, 0.1)
        noise = np.random.normal(0, std, (1024, 1024,3))
        a = a + noise
        a = np.clip(0, 1, a)
        return a

    @staticmethod
    def random_brightness(a):
        alpha = random.uniform(-0.2, 0.2)
        a = (1 + alpha) * a
        a = np.clip(0, 1, a)
        return a

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        # code for Linux
        img_name = img_path.split("/")[-1]
        # code for Window
        img_name = img_path.split("\\")[-1]
        label_path = f"{self.path}/label/{img_name}"

        img = skio.imread(img_path)
        img = img / 255
        # img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        label = cv2.cvtColor(cv2.imread(label_path), cv2.COLOR_BGR2GRAY)

        # data augmentation
        img, label = self.random_horizontal_flip(img, label)
        img, label = self.random_vertical_flip(img, label)
        img, label = self.random_rotate_90(img, label)
        img = self.random_gaussian(img)
        img = self.random_brightness(img)

        img = torch.from_numpy(img).permute(2, 0, 1).float()
        label = torch.from_numpy(label).unsqueeze(0).float()
        label = label / 255

        return img, label


class KmdfDataset_valid(torch.utils.data.Dataset):
    def __init__(self, path):
        self.path = path
        self.img_list = sorted(glob.glob(f"{path}/image/*"))
        # print(self.img_list)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        # code for Linux
        img_name = img_path.split("/")[-1]
        # code for Window
        img_name = img_path.split("\\")[-1]
        label_path = f"{self.path}/label/{img_name}"

        img = skio.imread(img_path)
        img = img / 255
        # img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        label = cv2.cvtColor(cv2.imread(label_path), cv2.COLOR_BGR2GRAY)

        img = torch.from_numpy(img).permute(2, 0, 1).float()
        label = torch.from_numpy(label).unsqueeze(0).float()
        label = label / 255

        return img, label


if __name__ == "__main__":
    testset = KmdfDataset("./dataset/220916_whole/test")
    data = next(iter(testset))
    img, label = data
    # print(img.shape)
    # save_image(img, '/home/nica/Downloads/img.png')
    # save_image(label, '/home/nica/Downloads/label.png')
