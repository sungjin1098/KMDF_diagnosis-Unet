'''
This code produces zero values as labels for normal patches
'''

import cv2
import numpy as np
import glob
import os
import skimage.io as skio


folder = '/home/nica/Downloads/KMDF_diagnosis-master/dataset/220822_whole/test/image/'
save_path = '/home/nica/Downloads/KMDF_diagnosis-master/dataset/220822_whole/test/label/'

os.makedirs(save_path, exist_ok = True)
files = glob.glob(folder + '*.jpg')

print('Starting...')

for file in files:
    print(file)
    img_name = file.split('/')[-1][:-4]
    img = skio.imread(file)
    img = np.uint8(np.zeros([img.shape[0], img.shape[1]]))
    cv2.imwrite(save_path + img_name + ".jpg", img)

print('Done!')