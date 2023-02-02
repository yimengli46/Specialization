import numpy as np
import matplotlib.pyplot as plt
import bz2
import _pickle as cPickle
import os
import glob
from modeling.utils.baseline_utils import create_folder
from PIL import Image
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib.patches as patches

scene_name = 'wcojb4TFT35_0'
split = 'val'
scene_name_with_index = '00802-wcojb4TFT35'
thresh = 0.2

img_folder = f'output/scene_traversal/{scene_name}'
output_folder = f'output/owl_detections'
saved_folder = f'{output_folder}/{scene_name}'

create_folder(saved_folder)

# ==================================== read semantic text file ===============================
env_name = scene_name[:-2]
semantic_file = f'data/scene_datasets/hm3d/{split}/{scene_name_with_index}/{env_name}.semantic.txt'

set_categories = set()
with open(f'{semantic_file}', "r") as reader:
    for idx, line in enumerate(reader.readlines()):
        if idx > 0:
            i_first_quotes = line.find('"')
            i_second_quotes = line.find('"', i_first_quotes+1)
            word = line[i_first_quotes+1:i_second_quotes]
            if 'wall' in word:
                continue
            if 'floor' in word:
                continue
            if 'ceiling' in word:
                continue
            if 'door' in word:
                continue
            set_categories.add(str(word))

'''
set_categories.remove('wall')
set_categories.remove('floor')
set_categories.remove('ceiling')
set_categories.remove('bath wall')
'''
#assert 1==2
# =================================== start detection ===============================
pbz_file_list = [os.path.splitext(os.path.basename(x))[0]
                 for x in sorted(glob.glob(f'{img_folder}/*.pbz2'))]

processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

texts = [list(set_categories)]
#texts = [['human face', 'rocket', 'nasa badge', 'star-spangled banner']]
#texts = [["a photo of a cat", "a photo of a dog"]]

for pbz_file in pbz_file_list[:-1]:
    with bz2.BZ2File(f'{img_folder}/{pbz_file}.pbz2', 'rb') as fp:
        print(f'pbz_file = {pbz_file}')
        npy_file = cPickle.load(fp)

        panor_img = np.concatenate(
            (npy_file[0], npy_file[1], npy_file[2], npy_file[3]), axis=1)

        panopSeg_list = []

        for i in range(4):
            img = npy_file[i]
            img = Image.fromarray(img)

            with torch.no_grad():
                inputs = processor(text=texts, images=img, return_tensors="pt")
                outputs = model(**inputs)
                target_sizes = torch.Tensor([img.size[::-1]])
                results = processor.post_process(
                    outputs=outputs, target_sizes=target_sizes)
                boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]
                text = texts[0]

                #fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                # ax.imshow(img)
                # ax.set_axis_off()

                fig = Figure()
                dpi = fig.get_dpi()
                fig.set_size_inches(
                    (512 + 1e-2) / dpi,
                    (512 + 1e-2) / dpi,
                )
                canvas = FigureCanvasAgg(fig)
                #ax = fig.gca()
                ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
                ax.axis("off")

                ax.imshow(img)

                for score, box, label in zip(scores, boxes, labels):
                    if score < thresh:
                        continue
                    print(
                        f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")

                    x1, y1, x2, y2 = box
                    rect = patches.Rectangle(
                        (x1, y1), x2-x1, y2-y1, linewidth=3, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)

                    ax.text(x1, y1, f'{texts[0][label]}: {score:1.2f}',
                                    ha='left', va='top', color='red',
                                    bbox={'facecolor': 'white', 'edgecolor': 'red'})

                # plt.show()
                s, (width, height) = canvas.print_to_buffer()
                buffer = np.frombuffer(s, dtype="uint8")
                img_rgba = buffer.reshape(height, width, 4)
                rgb, alpha = np.split(img_rgba, [3], axis=2)
                panopSeg_list.append(rgb.astype('uint8'))
                #assert 1==2

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
