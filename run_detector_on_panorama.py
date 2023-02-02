import numpy as np
import matplotlib.pyplot as plt
import bz2
import _pickle as cPickle
import os
import glob
from modeling.utils.baseline_utils import create_folder
from modeling.panoptic_prediction import PanopPred

scene_name = 'gmuS7Wgsbrx_0'

img_folder = f'output/scene_traversal/{scene_name}'
output_folder = f'output/detections'
saved_folder = f'{output_folder}/{scene_name}'

create_folder(saved_folder)

pbz_file_list = [os.path.splitext(os.path.basename(x))[0]
                 for x in sorted(glob.glob(f'{img_folder}/*.pbz2'))]

panop_pred = PanopPred()

for pbz_file in pbz_file_list[:-1]:
    with bz2.BZ2File(f'{img_folder}/{pbz_file}.pbz2', 'rb') as fp:
        print(f'pbz_file = {pbz_file}')
        npy_file = cPickle.load(fp)

        panor_img = np.concatenate(
            (npy_file[0], npy_file[1], npy_file[2], npy_file[3]), axis=1)

        panopSeg_list = []

        for i in range(4):
            img = npy_file[i]
            _, b = panop_pred.get_prediction(img, flag_vis=True)
            panopSeg_list.append(b.copy())
            # plt.imshow(b)
            # plt.title('b')
            # plt.show()
        panor_panopSeg = np.concatenate(
            (panopSeg_list[0], panopSeg_list[1], panopSeg_list[2], panopSeg_list[3]), axis=1)

        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(60, 20))
        ax[0].imshow(panor_img)
        ax[0].get_xaxis().set_visible(False)
        ax[0].get_yaxis().set_visible(False)
        ax[1].imshow(panor_panopSeg)
        ax[1].get_xaxis().set_visible(False)
        ax[1].get_yaxis().set_visible(False)
        fig.tight_layout()
        plt.title(f'panorama: {pbz_file}')
        # plt.show()
        fig.savefig(f'{saved_folder}/{pbz_file}.jpg', bbox_inches='tight')
        plt.close()

        #assert 1==2
