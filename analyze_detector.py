'''
evaluate the multiview classification model
with Detic detections and ground truth detections
'''

import numpy as np
import matplotlib.pyplot as plt
import bz2
import _pickle as cPickle
from sklearn.metrics import f1_score, average_precision_score
import pandas as pd
from dataloader_input_view_by_densely_sampled_locations import process_lvis_dict

bbox_type = 'Detic'  # 'Detic'
model_type = 'knowledge_graph'

# ================================= load the datasets ============================
hm3d_to_lvis_dict = np.load(
    'output/knowledge_graph/hm3d_to_lvis_dict.npy', allow_pickle=True).item()

# load the LVIS categories
with bz2.BZ2File('output/knowledge_graph/LVIS_categories_and_embedding.pbz2', 'rb') as fp:
    LVIS_dict = cPickle.load(fp)

goal_obj_list, goal_obj_index_list, lvis_cat_name_to_lvis_id_dict, lvis_id_to_lvis_cat_names_dict = process_lvis_dict(
    hm3d_to_lvis_dict, LVIS_dict)

# =============================== read the results =================================
with bz2.BZ2File(f'output/multilabel_pred_results_model_{model_type}_bbox_{bbox_type}.pbz2', 'rb') as fp:
    results_dict = cPickle.load(fp)

y_pred = results_dict['y_pred']
y_label = results_dict['y_label']
original_dist_all = results_dict['original_dist']
detector_pred_all = results_dict['detector_pred']

# ================= initialize data frame for visualization =======================
df = pd.DataFrame(columns=['Class_id', 'Categories', '# in the view', 'detector mAP',
                  'in: model pred mAP', '# near the view', 'near: model pred mAP'])
df['Categories'] = df['Categories'].astype(str)
df['# in the view'] = df['# in the view'].astype(int)
df['detector mAP'] = df['detector mAP'].astype(float)
df['in: model pred mAP'] = df['in: model pred mAP'].astype(float)
df['# near the view'] = df['# near the view'].astype(int)
df['near: model pred mAP'] = df['near: model pred mAP'].astype(float)

# ============================ analyze the data =========================
num_images, num_classes = y_pred.shape

for idx_class in range(num_classes):
    original_dist_class = original_dist_all[:, idx_class]

    label_in_the_view = (original_dist_class == 1)
    label_near_the_view = (original_dist_class == 2)

    # in the view
    # if label_in_the_view.sum() > 0:
    detector_class = detector_pred_all[:, idx_class]
    detector_mAP = average_precision_score(
        label_in_the_view, detector_class)

    pred_class = y_pred[:, idx_class]
    pred_mAP = average_precision_score(label_in_the_view, pred_class)
    # else:
    #     detector_mAP = 0
    #     pred_mAP = 0

    # out of the view
    # if label_near_the_view.sum() > 0:
    pred_mAP_near_view = average_precision_score(
        label_near_the_view, pred_class)
    # else:
    # pred_mAP_near_view = 0

    print(f'class_id = {idx_class}:')
    print(f'==> # in the view = {label_in_the_view.sum()}')
    print(f'==> detector mAP = {detector_mAP:.3f}')
    print(f'==> pred mAP = {pred_mAP:.3f}')
    print(f'==> # near the view = {label_near_the_view.sum()}')
    print(f'==> pred mAP = {pred_mAP_near_view:.3f}')
    print(
        f'==> synonyms = {lvis_id_to_lvis_cat_names_dict[goal_obj_index_list[idx_class]]}')

    df = df.append(
        {
            'Class_id': idx_class,
            'Categories': lvis_id_to_lvis_cat_names_dict[goal_obj_index_list[idx_class]],
            '# in the view': int(label_in_the_view.sum()),
            'detector mAP': detector_mAP,
            'in: model pred mAP': pred_mAP,
            '# near the view': int(label_near_the_view.sum()),
            'near: model pred mAP': pred_mAP_near_view
        },
        ignore_index=True)


# ======================= write results to html ===========================
html = df.to_html(float_format=lambda x: '%.3f' % x)
# write html to file
html_f = open(
    f'output/pandas_results/multilabel_pred_results_model_{model_type}_input_{bbox_type}_.html', "w")
html_f.write(f'<h5>{model_type} - {bbox_type}</h5>')
html_f.write(html)

df2 = df[['detector mAP', 'in: model pred mAP', 'near: model pred mAP']].mean()
html = df2.to_frame('mean').to_html(float_format=lambda x: '%.3f' % x)
html_f.write(f'<h5>Mean over all classes</h5>')
html_f.write(html)

html_f.close()
