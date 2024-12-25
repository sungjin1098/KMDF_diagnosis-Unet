import os
import cv2
import numpy as np
import shutil

# ## (Black label) generate black labels for Normal images
# save_img_path = './image_temp/'
# save_label_path = './label_temp/'
# img_dir = save_img_path
# paths = sorted(os.listdir(img_dir))
#
# for path in paths:
#     img_name = path[:-4]
#
#     # Load the image and generate the black label
#     img = cv2.imread(f'{img_dir}/{path}')
#     label = np.zeros_like(img)
#
#     # Save image and corresponding label in PNG format
#     cv2.imwrite(f'{save_img_path}/{img_name}.png', img)
#     cv2.imwrite(f'{save_label_path}/{img_name}.png', label)


# ### (Size check) Check if the size of the image is 1024x1024
# img_dir = './image_temp'
# paths =sorted(os.listdir(img_dir))
#
# for path in paths:
#     img = cv2.imread(f'{img_dir}/{path}')
#     h, w, _ = img.shape
#     if h != 1024 or w != 1024:
#         print(f'{img_dir}/{path}')
#         print(img.shape)
#         exit()


# ### (Resize) 2048을 1024로 단순 resizing. 쪼개지 않고.
# img_dir = './image_temp'
# label_dir = './label_temp'
# img_save_dir = './image_temp2'
# label_save_dir = './label_temp2'
#
# paths = sorted(os.listdir(img_dir))
#
# for path in paths:
#     img_name = path[:-4]
#
#     img = cv2.imread(f'{img_dir}/{path}')
#     img_resized = cv2.resize(img, (1024, 1024))
#     cv2.imwrite(f'{img_save_dir}/{img_name}.png', img_resized)
#
# paths = sorted(os.listdir(label_dir))
#
# for path in paths:
#     label_name = path[:-4]
#
#     label = cv2.imread(f'{label_dir}/{path}')
#     label_resized = cv2.resize(label, (1024, 1024))
#     cv2.imwrite(f'{label_save_dir}/{label_name}.png', label_resized)

################ Normal ################
# ### 이미지 파일명 앞에 P028_N_과 같은 이름 붙여주기
# name = 'P045_N_'
# img_dir = './image_temp'
# paths = sorted(os.listdir(img_dir))
#
# for path in paths:
#     os.rename(f'{img_dir}/{path}', f'{img_dir}/{name}{path}')


# ### 큰 이미지를 1024x1024 단위로 쪼개서 저장하기: Normal 이미지 버전
# img_dir = './image_temp'
# img_save_dir = './image_temp2'
# label_save_dir = './label_temp2'
#
# paths = sorted(os.listdir(img_dir))
#
#
# def gen_index(input_size, patch_size, overlap_size):
#     indices = []
#     for k in range(2):
#         z_range = list(range(0, input_size[k] - patch_size[k] + 1, overlap_size[k]))
#         if input_size[k] - patch_size[k] > z_range[-1]:
#             z_range.append(input_size[k] - patch_size[k])
#         indices.append(z_range)
#     return indices
#
#
# for path in paths:
#     img_name = path[:-4]
#     print(f'Processing... {img_name}')
#
#     img = cv2.imread(f'{img_dir}/{path}')
#     h = img.shape[0]
#     w = img.shape[1]
#
#     # patch size = 1024
#     patch = 1024
#
#     # patch 크기대로, 겹치지 않고 슬라이싱
#     val_ind = gen_index([h, w], [patch, patch], [patch, patch])
#
#     h_remain = h % patch if h % patch != 0 else patch
#     w_remain = w % patch if w % patch != 0 else patch
#
#     for hi in range(len(val_ind[0])):
#         for wi in range(len(val_ind[1])):
#             # sliced image: 맨 끝부분은 안쪽과 겹치도록 slicing
#             img_small = img[val_ind[0][hi]:val_ind[0][hi] + patch, val_ind[1][wi]:val_ind[1][wi] + patch, :]
#             index = hi*len(val_ind[1])+wi   # index: 좌측상단부터, 0, 1, 2, ...
#
#             cv2.imwrite(f'{img_save_dir}/{img_name}_{index}.png', img_small)
#
#             # label: patch 크기에 맞는 black 이미지
#             label_small = np.zeros((patch, patch))
#             cv2.imwrite(f'{label_save_dir}/{img_name}_{index}.png', label_small)


################ Tumor ################
# ### 이미지 파일명 앞에 P028_T_와 같은 이름 붙여주기 + Tumor에 붙는 _2 없애주기
# name = 'P046_T_'
# img_dir = './image_temp'
# label_dir = './label_temp'
#
# paths = sorted(os.listdir(img_dir))
# for path in paths:
#     img_name = path[:-4]
#     os.rename(f'{img_dir}/{path}', f'{img_dir}/{name}{path}')
#     os.rename(f'{label_dir}/{img_name}_2.png', f'{label_dir}/{name}{path}')


# ## Check the aspect ratio (TPM이미지와 label의 height, width 비율을 비교해보기)
# img_dir = './image_temp'
# label_dir = './label_temp'
# img_paths = sorted(os.listdir(img_dir))
#
# for img_path in img_paths:
#     print(img_path)
#     img = cv2.imread(f'{img_dir}/{img_path}')
#     h_i, w_i = img.shape[:2]
#     print(h_i/w_i, 'h', h_i, 'w', w_i, 'img')
#     label = cv2.imread(f'{label_dir}/{img_path}')
#     h, w = label.shape[:2]
#     print(h/w, 'h', h, 'w', w, 'label')
#     print('To match height: ', h_i*w/w_i, 'To match width: ', w_i*h/h_i)


# ### Label의 크기를 이미지 크기에 맞추기
# img_dir = './image_temp'
# label_dir = './label_temp'
#
# paths = sorted(os.listdir(img_dir))
# for path in paths:
#     print('Processing...', path)
#     img = cv2.imread(f'{img_dir}/{path}')
#     h, w = img.shape[:2]
#
#     label = cv2.imread(f'{label_dir}/{path}')
#     label_resized = cv2.resize(label, (w, h))
#     cv2.imwrite(f'{label_dir}/{path}', label_resized)
#
#     ### check
#     img = cv2.imread(f'{img_dir}/{path}')
#     label = cv2.imread(f'{label_dir}/{path}')
#     if img.shape[0] != label.shape[0] or img.shape[1] != label.shape[1]:
#         print('ERROR', path)
#         print('Image shape: ', img.shape)
#         print('Label shape: ', label.shape)
#         exit()


# ### 큰 이미지를 1024x1024 단위로 쪼개서 저장하기: Tumor 이미지 버전
# img_dir = './image_temp'
# label_dir = './label_temp'
# img_save_dir = './image_temp2'
# label_save_dir = './label_temp2'
#
# paths = sorted(os.listdir(img_dir))
#
#
# def gen_index(input_size, patch_size, overlap_size):
#     indices = []
#     for k in range(2):
#         z_range = list(range(0, input_size[k] - patch_size[k] + 1, overlap_size[k]))
#         if input_size[k] - patch_size[k] > z_range[-1]:
#             z_range.append(input_size[k] - patch_size[k])
#         indices.append(z_range)
#     return indices
#
#
# for path in paths:
#     img_name = path[:-4]
#     print(f'Processing... {img_name}')
#
#     img = cv2.imread(f'{img_dir}/{path}')
#     label = cv2.imread(f'{label_dir}/{path}')
#     h = img.shape[0]
#     w = img.shape[1]
#
#     # patch size = 1024
#     patch = 1024
#
#     # patch 크기대로, 겹치지 않고 슬라이싱
#     val_ind = gen_index([h, w], [patch, patch], [patch, patch])
#
#     h_remain = h % patch if h % patch != 0 else patch
#     w_remain = w % patch if w % patch != 0 else patch
#
#     for hi in range(len(val_ind[0])):
#         for wi in range(len(val_ind[1])):
#             # sliced image: 맨 끝부분은 안쪽과 겹치도록 slicing
#             img_small = img[val_ind[0][hi]:val_ind[0][hi] + patch, val_ind[1][wi]:val_ind[1][wi] + patch, :]
#             index = hi*len(val_ind[1])+wi   # index: 좌측상단부터, 0, 1, 2, ...
#             cv2.imwrite(f'{img_save_dir}/{img_name}_{index}.png', img_small)
#
#             label_small = label[val_ind[0][hi]:val_ind[0][hi] + patch, val_ind[1][wi]:val_ind[1][wi] + patch, :]
#             cv2.imwrite(f'{label_save_dir}/{img_name}_{index}.png', label_small)


# ## Normal patches in Tumor region 찾아내기.
# img_dir = './image_temp2'
# label_dir = './label_temp2'
# move_img_dir = './NinT_image'
# move_label_dir = './NinT_label'
# paths = sorted(os.listdir(img_dir))
#
# length = len(paths)
# for path in paths:
#     img_name = path[:-4]
#     label = cv2.imread(f'{label_dir}/{path}')
#
#     if np.sum(label) == 0:
#         print('Zero label...', path)
#         shutil.move(f'{img_dir}/{path}', f'{move_img_dir}/{path}')
#         shutil.move(f'{label_dir}/{path}', f'{move_label_dir}/{path}')


# ###### Trash #########

# ### Normal in Tumor patch면 NinT로 파일명 바꿔주기
# from tqdm import tqdm
# dir = './NinT_image'
# dir2 = './NinT_label'
# paths = sorted(os.listdir(dir))
# for path in tqdm(paths):
#     new = path[:5] + 'N' + path[6:-4] + '_NinT.png'
#     os.rename(f'{dir}/{path}', f'{dir}/{new}')
#     os.rename(f'{dir2}/{path}', f'{dir2}/{new}')


# ### 파일명에 공백이나 점 있으면 삭제하기
# dir = './KMDF_diagnosis_dataset/230703_whole_split/test'
# dirs = sorted(os.listdir(dir))
# for d in dirs:
#     full_dir = f'{dir}/{d}'
#     file_names = sorted(os.listdir(full_dir))
#     for file_name in file_names:
#         original = file_name
#         file_name = file_name.replace(" ", "_")
#         file_name = file_name[:-4].replace(".", "")
#         new = file_name + '.png'
#         os.rename(f'{full_dir}/{original}', f'{full_dir}/{new}')


# ### image와 label 파일명 같은지 확인
# img_dir = './KMDF_diagnosis_dataset/230703_whole_split/train/image'
# label_dir = './KMDF_diagnosis_dataset/230703_whole_split/train/label'
# img_paths = sorted(os.listdir(img_dir))
# label_paths = sorted(os.listdir(label_dir))
#
# cnt = 0
# for img_path in img_paths:
#     if img_path not in label_paths:
#         cnt += 1
#         print(img_path)
# print(cnt)
# print("======================")
# cnt = 0
# for label_path in label_paths:
#     if label_path not in img_paths:
#         cnt += 1
#         print(label_path)
# print(cnt)
