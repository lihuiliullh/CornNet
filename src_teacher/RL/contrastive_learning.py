import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision
import torch.distributed as dist
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F
import os
import pickle


class FeedForward(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeedForward, self).__init__()

        self.layer_1 = nn.Linear(input_dim, input_dim)
        self.layer_2 = nn.Linear(input_dim, int(input_dim * 0.5))
        self.layer_3 = nn.Linear(int(input_dim * 0.5), int(input_dim * 0.5))
        self.layer_out = nn.Linear(int(input_dim * 0.5), output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(int(input_dim))
        self.batchnorm2 = nn.BatchNorm1d(int(input_dim * 0.5))
        self.batchnorm3 = nn.BatchNorm1d(int(input_dim * 0.5))

    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_out(x)

        return x


class SimCLR(nn.Module):

    def __init__(self, input_dim, projection_dim, config):
        super(SimCLR, self).__init__()

        self.n_features = input_dim
        # only have one projector
        # the input is the action embedding
        # the goal is to make the embedding of different actions far from each other

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False),
        )

        self.nn_cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.config = config

    def forward(self, A):
        # do something about A
        x, y, z = A.shape
        B = A.view(x * y, z)
        loss = self.contrastive_learning(B)
        return loss

        # loss = 0
        # for i in range(A.shape[0]):
        #     data = A[i,:,:]
        #     loss += self.contrastive_learning(data)
        # return loss
    

    def do_proj(self, x_i, x_j):
        z_i = self.projector(x_i)
        z_j = self.projector(x_j)
        return z_i, z_j

    # generalize to 3D case, too lazy
    def info_nce_loss(self, z_i, z_j):
        features = torch.cat((z_i, z_j), dim=0)
        labels = torch.cat([torch.arange(z_i.shape[0]) for i in range(self.config.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.config.device)
        # normalize to sphere
        features = F.normalize(features, dim=1)
    
        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape
    
        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.config.gpu)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape
    
        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    
        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    
        logits = torch.cat([positives, negatives], dim=1)
        # the ground truth is index 0, so labels are all 0
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.config.gpu)
    
        logits = logits / self.config.temperature
        return logits, labels


    def contrastive_learning(self, input_data_feature):
        input_feature = F.normalize(input_data_feature, dim=1)
        # add gaussian noise
        variance = 0.13
        pos1 = input_feature + (variance**0.5)*torch.randn(input_feature.shape).to(self.config.gpu)
        pos2 = input_feature + (variance**0.5)*torch.randn(input_feature.shape).to(self.config.gpu)

        # generate two view ############
        # percentage = 0.8
        # l, w = input_feature.shape
        # total_ele = l * w
        # mask = np.ones(total_ele, dtype=int)
        # mask[int(total_ele * percentage):] = 0

        # np.random.shuffle(mask)
        # mask1 = torch.tensor(mask).view(l, w).to(args.gpu)
        # np.random.shuffle(mask)
        # mask2 = torch.tensor(mask).view(l, w).to(args.gpu)
        
        # pos1 = input_feature * mask1
        # pos2 = input_feature * mask2

        ####################################

        z_i, z_j = self.do_proj(pos1, pos2)
        # try to docompose
        logits, labels = self.info_nce_loss(z_i, z_j)
        loss = self.criterion(logits, labels)
        return loss
