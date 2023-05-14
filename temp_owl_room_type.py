''' 
run owl model to detect room types
'''
import torch
from PIL import Image
import bz2
import _pickle as cPickle
import matplotlib.pyplot as plt
import os
import glob
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import matplotlib.patches as patches

device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)

room_types = ["living room", "bathroom", 'dining room', 'kitchen',
              'bedroom', 'laundry room', 'pantry', 'office', 'garage', 'outdoor']
# text = clip.tokenize(room_types).to(device)
texts = [room_types]

processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
thresh = 0.2

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

    img = Image.fromarray(fron['rgb'])

    with torch.no_grad():
        inputs = processor(
            text=texts, images=img, return_tensors="pt")
        outputs = model(**inputs)
        target_sizes = torch.Tensor([img.size[::-1]])
        results = processor.post_process(
            outputs=outputs, target_sizes=target_sizes)
        boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]
        text = texts[0]

        # fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        # ax.imshow(img)
        # ax.set_axis_off()

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 15))
        ax.imshow(fron['rgb'])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

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

        plt.show()

    '''
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
    fig.savefig(f'output/OWL_room_type/{sample_name}.jpg')
    plt.close()
    '''
