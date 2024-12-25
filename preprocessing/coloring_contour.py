'''
This code provides segementation labels by coloring contour in patches 

Make sure that all contours are closed for proper operation

*reference link : https://stackoverflow.com/questions/44588279/find-and-draw-the-largest-contour-in-opencv-on-a-specific-color-python
'''

import cv2
import numpy as np
import glob
import os
import skimage.io as skio


# Setting directory path of input/output data

contour_path = '/home/nica/Downloads/220603_SNUBH_P029/P029_Tumor/image/'
save_path = '/home/nica/Downloads/220603_SNUBH_P029/P029_Tumor/label/'

os.makedirs(save_path, exist_ok=True)
files = glob.glob(contour_path + '*_2.png') # read label patches ('_2' : postfix of filename)

choice_colors = ['White', 'Blue']

for choice_color in choice_colors:
    for file in files:
        img = cv2.imread(file)
        img_name = file.split('/')[-1][:-4]
        print('------------')
        print(img_name)
        
        # Converting BGR to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Defining range of each color
        if choice_color == 'Blue':
            lower, upper = (24, 128, 115), (164, 255, 255)
        elif choice_color == 'White':
            lower, upper = (0, 0, 120), (0, 0, 255)
        else:
            raise Exception('Check color!')

        # Extracting choice color
        contour = cv2.inRange(hsv, lower, upper)

        # Finding all contours
        contours = cv2.findContours(contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        
        # Checking existence of contours
        try:
            big_contour = max(contours, key=cv2.contourArea)
        except:
            continue

        # Coloring contours on a black background
        result = np.zeros_like(contour)
        cv2.drawContours(result, [ctr for ctr in contours], -1, 255, cv2.FILLED)

        # Reversing pixel values in result for blue contours (blue contours wraps normal regions)
        if choice_color == 'Blue':
            temp = np.uint8(np.ones(result.shape) * 255)
            result = temp - result

        # Saving result
        cv2.imwrite(save_path + img_name + '_contour.jpg', contour)
        cv2.imwrite(save_path + img_name + '.jpg', result)