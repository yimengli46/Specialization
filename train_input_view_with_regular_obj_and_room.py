import torch.optim as optim
import os
import numpy as np
from modeling.utils.ResNet_multilabel import mlp, mlp_wo_room_type
from sseg_utils.saver import Saver
from sseg_utils.summaries import TensorboardSummary
import matplotlib.pyplot as plt
from dataloader_regular_obj_and_room import get_all_view_dataset, my_collate
import torch.utils.data as data
import torch
import torch.nn as nn
from core import cfg
import bz2
import _pickle as cPickle
import argparse
from sklearn.metrics import f1_score, average_precision_score
import random
import torch.nn.functional as F
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(1024),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                         (0.26862954, 0.26130258, 0.27577711)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                         (0.26862954, 0.26130258, 0.27577711)),
])


def gen_plot(y_pred, y_label):
    """Create a pyplot plot and save to buffer."""
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 2))

    y_pred = (np.array(y_pred).flatten() > 0.5)
    y_label = np.array(y_label).flatten()
    acc = (y_pred == y_label).mean()
    f1 = f1_score(y_label, y_pred, average='weighted')

    y_pred = np.reshape(y_pred, (-1, 6))
    y_label = np.reshape(y_label, (-1, 6))

    idx0 = random.choice((range(y_pred.shape[0])))
    y_pred = np.reshape(y_pred[idx0], (1, -1))
    y_label = np.reshape(y_label[idx0], (1, -1))

    ax[0].imshow(y_pred)
    ax[0].set_title(f'pred, acc = {acc:.2}, f1 = {f1:.2}')
    ax[1].imshow(y_label)
    ax[1].set_title('label')
    # buf = io.BytesIO()
    # plt.savefig(buf, format='png')
    # buf.seek(0)
    fig.tight_layout()
    return fig  # buf


def train(model_type):
    # model_type = 'knowledge_graph'  # 'knowledge_graph'  # 'context_matrix'  # 'resnet'

    if model_type == 'mlp':
        cfg.merge_from_file(
            'configs/exp_train_input_view_model_MLP.yaml')
    elif model_type == 'mlp_wo_room_type':
        cfg.merge_from_file(
            'configs/exp_train_input_view_model_MLP_wo_room_type.yaml')
    elif model_type == 'mlp_visible_only':
        cfg.merge_from_file(
            'configs/exp_train_input_view_model_MLP_visible_only.yaml')
    else:
        raise NotImplementedError("This model is not implemented.")
    cfg.freeze()

    output_folder = cfg.PRED.VIEW.SAVED_FOLDER
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    saver = Saver(output_folder)

    cfg.dump(stream=open(
        f'{saver.experiment_dir}/experiment_config.yaml', 'w'))

    # ============================================ Define Tensorboard Summary =================================
    summary = TensorboardSummary(saver.experiment_dir)
    writer = summary.create_summary()

    # ============================================== load necessary data ============================
    hm3d_to_lvis_dict = np.load(
        'output/knowledge_graph/hm3d_to_lvis_dict.npy', allow_pickle=True).item()

    # load the LVIS categories
    with bz2.BZ2File(f'output/knowledge_graph/LVIS_categories_and_embedding.pbz2', 'rb') as fp:
        LVIS_dict = cPickle.load(fp)
        lvis_cat_synonyms_list = LVIS_dict['cat_synonyms']
        lvis_rowid_to_catid_dict = LVIS_dict['rowid2catid_dict']
        lvis_cat_synonyms_embedding = LVIS_dict['cat_synonyms_embedding']
        lvis_cat_index_embedding = LVIS_dict['cat_id_embedding']

    # get index of all the goal objs
    goal_obj_list = sorted(list(set(hm3d_to_lvis_dict.values())))  # size: 351
    goal_obj_index_list = list(set(lvis_rowid_to_catid_dict[lvis_cat_synonyms_list.index(
        cat_syn)] for cat_syn in goal_obj_list))  # size: 310
    goal_obj_index_embeddings = lvis_cat_index_embedding[[
        i - 1 for i in goal_obj_index_list]]  # shape: 310 x 384

    # =========================================================== Define Dataloader ==================================================
    data_folder = cfg.PRED.VIEW.DENSELY_SAMPLED_LOCATIONS_SAVED_FOLDER

    dataset_train = get_all_view_dataset(
        'train', data_folder, hm3d_to_lvis_dict, LVIS_dict, train_transform)
    dataloader_train = data.DataLoader(dataset_train,
                                       batch_size=cfg.PRED.VIEW.BATCH_SIZE,
                                       num_workers=cfg.PRED.VIEW.NUM_WORKERS,
                                       shuffle=True,
                                       collate_fn=my_collate
                                       )

    dataset_val = get_all_view_dataset(
        'val', data_folder, hm3d_to_lvis_dict, LVIS_dict, test_transform)
    dataloader_val = data.DataLoader(dataset_val,
                                     batch_size=cfg.PRED.VIEW.BATCH_SIZE,
                                     num_workers=cfg.PRED.VIEW.NUM_WORKERS,
                                     shuffle=False,
                                     collate_fn=my_collate,
                                     )

    # ================================================================================================================================

    # Define network
    if cfg.PRED.VIEW.MODEL_TYPE == 'mlp':
        model = mlp()
    elif cfg.PRED.VIEW.MODEL_TYPE == 'mlp_wo_room_type':
        model = mlp_wo_room_type()
    model = nn.DataParallel(model)
    model = model.cuda()

    # =========================================================== Define Optimizer ================================================
    train_params = [{'params': model.parameters(), 'lr': cfg.PRED.VIEW.LR}]
    optimizer = optim.Adam(
        train_params, lr=cfg.PRED.VIEW.LR, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Define Criterion
    # whether to use class balanced weights
    weight = None

    criterion = nn.BCEWithLogitsLoss()
    best_test_loss = 1e10

    # ===================================================== Resuming checkpoint ====================================================
    best_pred = 0.0
    if cfg.PRED.VIEW.RESUME != '':
        if not os.path.isfile(cfg.PRED.VIEW.RESUME):
            raise RuntimeError(
                "=> no checkpoint found at '{}'" .format(cfg.PRED.VIEW.RESUME))
        checkpoint = torch.load(cfg.PRED.VIEW.RESUME)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_pred = checkpoint['best_pred']
        print("=> loaded checkpoint '{}' (epoch {})".format(
            cfg.PRED.VIEW.RESUME, checkpoint['epoch']))

    # =================================================================trainin
    for epoch in range(cfg.PRED.VIEW.EPOCHS):
        train_loss = 0.0
        model.train()
        iter_num = 0

        for batch in dataloader_train:
            print('epoch = {}, iter_num = {}'.format(epoch, iter_num))
            inputs, gt_outputs, batch_de_or_vi = batch['input'], batch['output'], batch['detected_or_vicinity']
            print(f'inputs.shape = {inputs.shape}')
            print(f'gt_outputs.shape = {gt_outputs.shape}')

            A, B, _ = inputs.shape
            inputs = inputs.view(A * B, -1)
            gt_outputs = gt_outputs.view(A * B)
            batch_de_or_vi = batch_de_or_vi.view(A * B)

            inputs = inputs.cuda()
            gt_outputs = gt_outputs.cuda()
            batch_de_or_vi = batch_de_or_vi.cuda()

            if cfg.PRED.VIEW.MULTILABEL_MODE == 'visible_only':
                mask = (batch_de_or_vi == 0) | (batch_de_or_vi == 2)
                inputs = inputs[mask, :]
                gt_outputs = gt_outputs[mask]

            if inputs.shape[0] == 0:
                continue

            # ================================================ compute loss =============================================
            if cfg.PRED.VIEW.MODEL_TYPE == 'mlp':
                output = model(inputs)  # batchsize x 1 x H x W
            elif cfg.PRED.VIEW.MODEL_TYPE == 'mlp_wo_room_type':
                output = model(inputs)  # batchsize x 1 x H x W
            print(f'output.shape = {output.shape}')
            # print(f'output = {output}')
            # print(f'dists.shape = {dists.shape}')
            # output = output.view(-1)
            # dists = dists.view(-1)

            loss = criterion(output, gt_outputs.float())
            # print(f'dists = {dists.nonzero()}')

            # ================================================= compute gradient =================================================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            print('Train loss: %.3f' % (train_loss / (iter_num + 1)))
            writer.add_scalar('train/total_loss_iter', loss.item(),
                              iter_num + len(dataloader_train) * epoch)

            iter_num += 1

        writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' %
              (epoch, iter_num * cfg.PRED.VIEW.BATCH_SIZE))
        print('Loss: %.3f' % train_loss)

    # ======================================================== evaluation stage =====================================================

        if epoch % cfg.PRED.VIEW.EVAL_INTERVAL == 0:
            model.eval()
            test_loss = []
            iter_num = 0

            with torch.no_grad():
                y_pred = np.zeros((0, 6))
                y_label = np.zeros((0, 6))
                y_de_or_vi = np.zeros((0, 6))

                sampled_batch_ids = random.choices(
                    range(len(dataloader_val)), k=10)
                count_vis_img = 0

                for idx_dl, batch in enumerate(dataloader_val):
                    print('epoch = {}, iter_num = {}'.format(epoch, iter_num))
                    inputs, gt_outputs, batch_de_or_vi = batch['input'], batch['output'], batch['detected_or_vicinity']
                    print(f'inputs.shape = {inputs.shape}')
                    print(f'gt_outputs.shape = {gt_outputs.shape}')

                    A, B, _ = inputs.shape
                    inputs = inputs.view(A * B, -1)
                    gt_outputs = gt_outputs.view(A * B)

                    inputs = inputs.cuda()
                    gt_outputs = gt_outputs.cuda()

                    '''
                    if cfg.PRED.VIEW.MULTILABEL_MODE == 'visible_only':
                        mask = (batch_de_or_vi == 0) | (batch_de_or_vi == 2)
                        inputs = inputs[mask, :]
                        gt_outputs = gt_outputs[mask]

                    if inputs.shape[0] == 0:
                        continue
                    '''

                    # ================================================ compute loss =============================================

                    if cfg.PRED.VIEW.MODEL_TYPE == 'mlp':
                        # batchsize x 1 x H x W
                        output = model(inputs)
                    elif cfg.PRED.VIEW.MODEL_TYPE == 'mlp_wo_room_type':
                        output = model(inputs)  # batchsize x 1 x H x W
                    print(f'output.shape = {output.shape}')

                    loss = criterion(output, gt_outputs.float())

                    # concatenate the results
                    output = output.view(A, B).cpu().numpy()
                    gt_outputs = gt_outputs.view(A, B).cpu().numpy()
                    batch_de_or_vi = batch_de_or_vi.numpy()

                    y_pred = np.concatenate((y_pred, output), axis=0)
                    y_label = np.concatenate((y_label, gt_outputs), axis=0)
                    y_de_or_vi = np.concatenate((y_de_or_vi, batch_de_or_vi), axis=0)

                    '''
                    if idx_dl in sampled_batch_ids:
                        output = np.array(output)
                        gt_outputs = np.array(gt_outputs)
                        fig = gen_plot(output, gt_outputs)
                        writer.add_figure(
                            f'val/sampled_result_{count_vis_img}', fig)
                        count_vis_img += 1
                    '''

                    test_loss.append(loss.item())
                    print(f'Test loss: {test_loss[-1]:.3f}')
                    writer.add_scalar('val/total_loss_iter', test_loss[-1],
                                      iter_num + len(dataloader_val) * epoch)

                    iter_num += 1

            # Fast test during the training
            test_loss = np.mean(test_loss)
            writer.add_scalar('val/total_loss_epoch',
                              test_loss, epoch)

            aps = []
            aps_vi = []
            for idx_class in range(0, y_label.shape[1]):
                class_label = y_label[:, idx_class]
                class_pred = y_pred[:, idx_class]
                class_de_or_vi = y_de_or_vi[:, idx_class]
                if class_label.sum() > 0:
                    mask = (class_de_or_vi == 0) | (class_de_or_vi == 3)
                    ap = average_precision_score(
                        class_label[mask], class_pred[mask])
                    print(
                        f'-------  Class: {idx_class}     AP: {ap:.4f}  -------')
                    aps.append(ap)

                    mask = (class_de_or_vi == 0) | (class_de_or_vi == 2)
                    ap_vi = average_precision_score(
                        class_label[mask], class_pred[mask])
                    print(
                        f'-------vi  Class: {idx_class}     AP: {ap_vi:.4f}  -------')
                    aps_vi.append(ap_vi)

            mAP = np.mean(aps)
            mAP_vi = np.mean(aps_vi)
            writer.add_scalar('val/mAP', mAP, epoch)
            writer.add_scalar('val/mAP_vi', mAP_vi, epoch)

            print('Validation:')
            print(f'Epoch: {epoch}, numImages: {len(dataloader_val)}')
            print(f'Loss: {test_loss:.3f}, mAP: {mAP:.3f}, mAP_vi: {mAP_vi:.3f}')

            saver.save_checkpoint({
                # 'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                # 'optimizer': optimizer.state_dict(),
                # 'loss': test_loss,
            }, filename='checkpoint.pth.tar')

            if test_loss < best_test_loss:
                best_test_loss = test_loss

                saver.save_checkpoint({
                    # 'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    # 'optimizer': optimizer.state_dict(),
                    # 'loss': test_loss,
                }, filename='best_checkpoint.pth.tar')

        scheduler.step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, default='mlp')
    args = parser.parse_args()

    train(args.model)


if __name__ == "__main__":
    main()
