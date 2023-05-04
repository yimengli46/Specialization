import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from core import cfg
import bz2
import _pickle as cPickle
import scipy.sparse as sp
import torch.nn.functional as F
import clip


class resnet(nn.Module):
    def __init__(
        self,
        n_channel_in,
        n_class_out,
        lvis_cat_synonyms_list, lvis_cat_synonyms_embedding,
        goal_obj_index_list, goal_obj_index_embeddings
    ):
        super().__init__()

        self.resnet = models.resnet18(pretrained=False, num_classes=256)

        # Adapted resnet from:
        # https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
        self.resnet.conv1 = nn.Conv2d(
            n_channel_in, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.fc1 = nn.Linear(256 + 384, 256)
        self.fc2 = nn.Linear(256, 1)

        self.lvis_cat_synonyms_list = lvis_cat_synonyms_list
        self.lvis_cat_synonyms_embedding = lvis_cat_synonyms_embedding
        self.goal_obj_index_embeddings = torch.tensor(
            goal_obj_index_embeddings).float()
        self.goal_obj_index_list = goal_obj_index_list

    def forward(self, x, target_obj_list):
        B = x.shape[0]

        x = self.resnet(x)  # B x 256
        # print(f'x.shape = {x.shape}')

        target_embedding = np.zeros((B, 310, 384), dtype=np.float32)
        for idx0, target_obj in enumerate(target_obj_list):
            for idx1, target in enumerate(target_obj):
                target_embedding[idx0, idx1] = self.lvis_cat_synonyms_embedding[self.lvis_cat_synonyms_list.index(
                    target)]
        target_embedding = torch.tensor(
            target_embedding).float().cuda()  # B x 310 x 384

        # encode the target
        # target_embedding = np.stack([self.lvis_cat_synonyms_embedding[self.lvis_cat_synonyms_list.index(
        #     target_obj)] for target_obj in target_obj_list])  # B x 384
        # target_embedding = torch.tensor(
        #     target_embedding).float().cuda()  # B x 384
        # print(f'target_embedding.shape = {target_embedding.shape}')

        # z = torch.cat((z, target_embedding), dim=1)

        x = x.view(B, 1, -1)
        z = torch.cat((x.expand(B, 310, 256), target_embedding), dim=2)

        z = F.relu(self.fc1(z))
        y_pred = self.fc2(z)
        # print(f'y_pred.shape = {y_pred.shape}')
        return y_pred


class context_matrix(nn.Module):
    def __init__(self, lvis_cat_synonyms_list, lvis_cat_synonyms_embedding,
                 goal_obj_index_list, goal_obj_index_embeddings):
        super().__init__()

        self.n = len(goal_obj_index_list)
        self.lvis_cat_synonyms_list = lvis_cat_synonyms_list
        self.lvis_cat_synonyms_embedding = lvis_cat_synonyms_embedding
        self.goal_obj_index_embeddings = torch.tensor(
            goal_obj_index_embeddings).float().detach()  # 1 x 310 x 384
        self.goal_obj_index_list = goal_obj_index_list

        self.fc1 = nn.Linear(310 * 6, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, objbb_list, target_obj_list):
        B = len(objbb_list)
        objstate = torch.zeros(B, self.n, 5).float()
        # compute the first 4 columns of the context matrix
        for idx, objbb in enumerate(objbb_list):
            for bbox in objbb:
                x1, y1, x2, y2, cat_id = bbox
                ind = self.goal_obj_index_list.index(cat_id)
                objstate[idx][ind][0] = 1
                objstate[idx][ind][1] = (
                    x1 / 2. + x2 / 2.) / cfg.SENSOR.OBS_WIDTH
                objstate[idx][ind][2] = (
                    y1 / 2. + y2 / 2.) / cfg.SENSOR.OBS_WIDTH
                objstate[idx][ind][3] = abs(x2 - x1) * abs(y2 - y1) / \
                    (cfg.SENSOR.OBS_WIDTH * cfg.SENSOR.OBS_WIDTH)
                objstate[idx][ind][4] = 1
        objstate = objstate.unsqueeze(
            1).expand(-1, self.n, -1, -1)  # B x 310 x 310 x 5

        # compute the last column of the context matrix
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        embedding_similarity = torch.zeros(
            B, self.n, self.n).float()  # B x 310 x 310
        for idx in range(B):
            target_embedding = np.stack([self.lvis_cat_synonyms_embedding[self.lvis_cat_synonyms_list.index(
                target_obj)] for target_obj in target_obj_list[idx]])  # 310 x 384
            target_embedding = torch.tensor(
                target_embedding).float()  # 310 x 384
            goal_obj_embedding_similarity = cos(self.goal_obj_index_embeddings,
                                                target_embedding).float()
            embedding_similarity[idx] = goal_obj_embedding_similarity
            # embedding_similarity[idx] = torch.eye(self.n).float()
        # print(
        #     f'self.goal_obj_index_embeddings.shape = {self.goal_obj_index_embeddings.shape}')
        # print(f'embedding_sim.shape = {embedding_similarity.shape}')

        objstate = torch.cat(
            (objstate, embedding_similarity.unsqueeze(3)), dim=3)  # B x 310 x 310 x 6
        # print(f'objstate.shape = {objstate.shape}')

        B, num_classes = objstate.shape[:2]
        x = objstate.view(B * num_classes, -1).cuda()
        # print(f'x.shape = {x.shape}')

        x = F.relu(self.fc1(x))
        y_pred = self.fc2(x)
        return y_pred


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    A = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    A = A.tocsr().toarray()
    return A


class knowledge_graph(nn.Module):
    def __init__(self, lvis_cat_synonyms_list, lvis_cat_synonyms_embedding,
                 goal_obj_index_list, goal_obj_index_embeddings):
        super().__init__()

        # get and normalize adjacency matrix
        with bz2.BZ2File('output/knowledge_graph/LVIS_relationships.pbz2', 'rb') as fp:
            LVIS_relationships = cPickle.load(fp)
            adjacency_cat_id = LVIS_relationships['adjacency_cat_id']
            all_obj_cat_ids = LVIS_relationships['all_obj_cat_ids']

        self.relationships_all_obj_cat_ids = all_obj_cat_ids

        adjacency_cat_id = adjacency_cat_id.astype('float32')
        adjacency_cat_id += 1e-5
        A = normalize_adj(adjacency_cat_id)
        # torch.nn.Parameter(torch.tensor(A))
        self.A = torch.tensor(A).detach().cuda()

        self.n = len(goal_obj_index_list)
        self.lvis_cat_synonyms_list = lvis_cat_synonyms_list
        self.lvis_cat_synonyms_embedding = lvis_cat_synonyms_embedding
        self.goal_obj_index_embeddings = torch.tensor(
            goal_obj_index_embeddings).float().detach()  # 310 x 384
        self.goal_obj_index_list = goal_obj_index_list

        self.W0 = nn.Linear(694, 694, bias=False)
        self.W1 = nn.Linear(694, 694, bias=False)
        self.W2 = nn.Linear(694, 5, bias=False)
        self.W3 = nn.Linear(10, 1, bias=False)

        self.fc1 = nn.Linear(self.n, 310)
        self.fc2 = nn.Linear(310, 1)

    def forward(self, objbb_list, target_obj_list):
        B = len(objbb_list)
        class_onehot = torch.zeros(B, 1, self.n).float()  # B x 1 x n

        for idx, objbb in enumerate(objbb_list):
            for bbox in objbb:
                x1, y1, x2, y2, cat_id = bbox
                ind = self.goal_obj_index_list.index(cat_id)
                class_onehot[idx, 0, ind] = 1

        class_node_embedding = torch.cat((class_onehot.expand(
            -1, self.n, -1), self.goal_obj_index_embeddings.unsqueeze(0).expand(B, -1, -1)), dim=2).cuda()  # B x 310 x 694
        # print(f'class_node_embedding.shape = {class_node_embedding.shape}')

        x = torch.bmm(self.A.unsqueeze(0).expand(B, -1, -1),
                      class_node_embedding)  # B x 310 x 694
        # print(f'x.shape = {x.shape}')
        x = F.relu(self.W0(x))
        x = torch.bmm(self.A.unsqueeze(0).expand(B, -1, -1), x)
        x = F.relu(self.W1(x))
        x = torch.bmm(self.A.unsqueeze(0).expand(B, -1, -1), x)
        x = F.relu(self.W2(x))  # B x 310 x 5
        # print(f'x.shape = {x.shape}')

        # ================ build objstate
        objstate = torch.zeros(B, self.n, 4).float()
        # compute the first 4 columns of the context matrix
        for idx, objbb in enumerate(objbb_list):
            for bbox in objbb:
                x1, y1, x2, y2, cat_id = bbox
                ind = self.goal_obj_index_list.index(cat_id)
                objstate[idx][ind][0] = 1
                objstate[idx][ind][1] = (
                    x1 / 2. + x2 / 2.) / cfg.SENSOR.OBS_WIDTH
                objstate[idx][ind][2] = (
                    y1 / 2. + y2 / 2.) / cfg.SENSOR.OBS_WIDTH
                objstate[idx][ind][3] = abs(x2 - x1) * abs(y2 - y1) / \
                    (cfg.SENSOR.OBS_WIDTH * cfg.SENSOR.OBS_WIDTH)
        objstate = objstate.unsqueeze(
            1).expand(-1, self.n, -1, -1)  # B x 310 x 310 x 4

        # compute the last column of the context matrix
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        embedding_similarity = torch.zeros(
            B, self.n, self.n).float()  # B x 310 x 310
        for idx in range(B):
            target_embedding = np.stack([self.lvis_cat_synonyms_embedding[self.lvis_cat_synonyms_list.index(
                target_obj)] for target_obj in target_obj_list[idx]])  # 310 x 384
            target_embedding = torch.tensor(
                target_embedding).float()  # 310 x 384
            goal_obj_embedding_similarity = cos(self.goal_obj_index_embeddings,
                                                target_embedding).float()
            embedding_similarity[idx] = goal_obj_embedding_similarity
            # embedding_similarity[idx] = torch.eye(self.n).float()
        # print(
        #     f'self.goal_obj_index_embeddings.shape = {self.goal_obj_index_embeddings.shape}')
        # print(f'embedding_sim.shape = {embedding_similarity.shape}')

        objstate = torch.cat(
            (objstate, embedding_similarity.unsqueeze(3)), dim=3).cuda()  # B x 310 x 310 x 5

        # ================== concatenate the two
        x = x.unsqueeze(1).expand(-1, self.n, -1, -1)  # B x 310 x 310 x 5
        x = torch.cat((x, objstate), dim=2)  # B x 310 x 310 x 10
        x = x.view(B * self.n, self.n, -1)
        # print(f'x.shape = {x.shape}')
        x = torch.bmm(self.A.unsqueeze(0).expand(B * self.n, -1, -1), x)
        x = F.relu(self.W3(x))  # (B x 310) x 310 x 1
        x = x.view(B * self.n, -1)

        x = F.relu(self.fc1(x))
        y_pred = self.fc2(x)

        return y_pred


class clip_fc(nn.Module):
    def __init__(self):
        super().__init__()

        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")

        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, image, target_obj_list):
        # image = self.preprocess(image).cuda()
        text = clip.tokenize(target_obj_list).cuda()
        # print(f'image.shape = {image.shape}')
        # print(f'text.shape = {text.shape}')

        with torch.no_grad():
            image_features = self.model.encode_image(image).float()
            text_features = self.model.encode_text(text).float()

        # print(f'image_features.shape = {image_features.shape}')
        # print(f'text_features.shape = {text_features.shape}')
        z = torch.cat((image_features, text_features), dim=1)
        # print(f'z.shape = {z.shape}')
        z = F.relu(self.fc1(z))
        y_pred = self.fc2(z)

        return y_pred
