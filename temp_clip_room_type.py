'''
run clip to get room types
'''
import torch
import clip
from PIL import Image
import bz2
import _pickle as cPickle
import matplotlib.pyplot as plt
import os
import glob

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

room_types = ["a living room", "a bathroom", "a recreation room",
              'a dining room', 'a kitchen', 'a bedroom', 'a laundry room', 'a pantry']
text = clip.tokenize(room_types).to(device)

data_folder = 'output/training_data_input_view_by_densely_sample_locations'
scene_name = '00009-vLpv2VX547B'
floor_id = '0'

sample_name_list = [os.path.splitext(os.path.basename(x))[0]
                    for x in sorted(glob.glob(f'{data_folder}/train/{scene_name}_{floor_id}/*.pbz2'))]
print(f'find {len(sample_name_list)} files.')

sample_name = '00000'
for sample_name in sample_name_list:
    print(f'sample_name = {sample_name}')
    with bz2.BZ2File(f'{data_folder}/train/{scene_name}_{floor_id}/{sample_name}.pbz2', 'rb') as fp:
        fron = cPickle.load(fp)

    # rgb_img = Image.fromarray(fron['rgb'])
    image = preprocess(Image.fromarray(fron['rgb'])).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

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
    fig.savefig(f'output/CLIP_room_type/{sample_name}.jpg')
    plt.close()
