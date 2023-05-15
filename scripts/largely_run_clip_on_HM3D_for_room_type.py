'''
run clip to get room types on all the scenes
'''
import torch
import clip
from PIL import Image
import bz2
import _pickle as cPickle
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

room_types = ['a living room', 'a bathroom', 'a dining room', 'a kitchen',
              'a bedroom', 'a pantry', 'an office', 'a garage', 'outdoor', 'a corridor']
text = clip.tokenize(room_types).to(device)

# ========================== create folders =====================
data_folder = '../Specialization/output/training_data_input_view_by_densely_sample_locations'
output_folder = '../Specialization/output/CLIP_room_type'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

split = 'train'  # 'val'
saved_folder = f'{output_folder}/{split}'
if not os.path.exists(saved_folder):
    os.mkdir(saved_folder)

# =============================================================
# ================ load the scenes =====================
scene_list = sorted(next(os.walk(f'{data_folder}/{split}'))[1])
scene_list = scene_list

for scene_floor in scene_list:
    print(f'scene_floor = {scene_floor}')
    scene_name, floor_id = scene_floor.split('_')

    scene_saved_folder = f'{saved_folder}/{scene_floor}'
    if not os.path.exists(scene_saved_folder):
        os.mkdir(scene_saved_folder)

    sample_name_list = [os.path.splitext(os.path.basename(x))[0]
                        for x in sorted(glob.glob(f'{data_folder}/{split}/{scene_name}_{floor_id}/*.pbz2'))]
    for sample_name in sample_name_list:
        with bz2.BZ2File(f'{data_folder}/{split}/{scene_name}_{floor_id}/{sample_name}.pbz2', 'rb') as fp:
            fron = cPickle.load(fp)
        print(f'sample_name = {sample_name}')

        image = preprocess(Image.fromarray(
            fron['rgb'])).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)

            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

        np.save(f'{scene_saved_folder}/{sample_name}_clip_room_types.npy', probs)

        print("Label probs:", probs)
        prob_str = ""
        for i_type in range(len(room_types)):
            prob_str += f'{room_types[i_type]}: {probs[i_type]:.3f}, '
            if i_type == 4:
                prob_str += '\n'

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 15))
        ax.imshow(fron['rgb'])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title(prob_str)
        # fig.tight_layout()
        # plt.show()
        fig.savefig(f'{scene_saved_folder}/{sample_name}_clip_room_types.jpg')
        plt.close()
