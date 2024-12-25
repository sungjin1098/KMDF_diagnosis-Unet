import matplotlib.pyplot as plt
import random


def gen_index(input_size, patch_size, overlap_size):
    # B, C, W, H = input_size
    indices = []
    for k in range(2):
        z_range = list(range(0, input_size[k] - patch_size[k] + 1, overlap_size[k]))
        if input_size[k] - patch_size[k] > z_range[-1]:
            z_range.append(input_size[k] - patch_size[k])
        indices.append(z_range)
    return indices


def gen_index_random(input_size, limit):
    # B, C, H, W = input_size
    indices = []
    # extract 4 patches at once
    for k in range(2):
        z_range = []
        for _ in range(1):
            z_range.append(random.randint(0, input_size[k] - limit))
        indices.append(z_range)
    return indices


# def draw_loss_curve(train, valid, name, save_curve):
#     plt.clf()
#     x = [i for i in range(1, train.shape[0] + 1)]
#     plt.plot(x, train, lw=0.75, color='red', label='train_loss')
#     plt.plot(x, valid, lw=0.75, color='blue', label='valid_loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend(loc='upper left', prop={'size': 6})
#     plt.savefig(save_curve + '/%s.png'%(name))
