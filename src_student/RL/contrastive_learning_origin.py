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

    def __init__(self, input_dim, n_features, projection_dim):
        super(SimCLR, self).__init__()

        self.encoder = FeedForward(input_dim, n_features)
        self.n_features = n_features

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False),
        )

    def forward(self, x_i, x_j, orig):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        h_i = h_i + orig
        h_j = h_j + orig

        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        return h_i, h_j, z_i, z_j
    
    def get_embedding(self, x):
        h = self.encoder(x)
        return h + x

nn_cos = nn.CosineSimilarity(dim=1, eps=1e-6)
def info_nce_loss(z_i, z_j, args):
    features = torch.cat((z_i, z_j), dim=0)
    labels = torch.cat([torch.arange(z_i.shape[0]) for i in range(args.n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(args.gpu)
    # normalize to sphere
    features = F.normalize(features, dim=1)
 
    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape
 
    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(args.gpu)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape
 
    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
 
    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
 
    logits = torch.cat([positives, negatives], dim=1)
    # the ground truth is index 0, so labels are all 0
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(args.gpu)
 
    logits = logits / args.temperature
    return logits, labels


def sim_regular(hi, hj, org):
    #similarity_matrix = torch.matmul(hi, hj.T)
    #similarity_matrix2 = torch.matmul(org, org.T)
    hi = F.normalize(hi, dim=1)
    hj = F.normalize(hj, dim=1)
    org = F.normalize(org, dim=1)

    similarity_matrix1 = torch.matmul(hi, hi.T)
    similarity_matrix2 = torch.matmul(hj, hj.T)
    similarity_matrix3 = torch.matmul(org, org.T)

    distance_ = similarity_matrix3 - similarity_matrix1
    distance_ = torch.square(distance_)
    d1 = torch.mean(distance_)

    distance_ = similarity_matrix3 - similarity_matrix2
    distance_ = torch.square(distance_)
    d2 = torch.mean(distance_)

    return d1 + d2

def contrastive_learning(input_data_feature, args):
    input_data_feature = F.normalize(input_data_feature, dim=1)
    input_data_feature.requires_grad = False
    #input_data_feature = input_data_feature * 100
    model = SimCLR(input_data_feature.shape[1], input_data_feature.shape[1], args.hidden_dim)
    model.to(args.gpu)
    #criterion = NT_Xent(args.batch_size, args.temperature, args.world_size)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.decay_rate:
        scheduler = ExponentialLR(optimizer, args.decay_rate)
    #############
    unified_feature = input_data_feature
    train_idx = list(range(input_data_feature.shape[0]))
    best_loss = 9999999

    for it in range(1, args.max_epoch+1):
        loss_epoch = 0
        np.random.shuffle(train_idx)

        for j in tqdm(range(0, len(train_idx), args.batch_size)):
            
            data_idx = train_idx[j:j+args.batch_size]
            input_feature = unified_feature[data_idx]

            variance = 0.13
            pos1 = input_feature + (variance**0.5)*torch.randn(input_feature.shape).to(args.gpu)
            pos2 = input_feature + (variance**0.5)*torch.randn(input_feature.shape).to(args.gpu)

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
            
            pos1 = pos1.detach()
            pos2 = pos2.detach()

            optimizer.zero_grad()
            h_i, h_j, z_i, z_j = model(pos1, pos2, input_feature.detach())
            # try to docompose
            logits, labels = info_nce_loss(z_i, z_j, args)
            loss = criterion(logits, labels)

            loss.backward(retain_graph=True)
            optimizer.step()

            loss_epoch += loss.item()
            #loss_epoch -= loss2.item()
            a = 0
        if args.decay_rate:
            scheduler.step()
        print(loss_epoch)
        if loss_epoch < best_loss:
            best_loss = loss_epoch
            # save model
            torch.save(model.state_dict(), os.path.join(args.state_dir, "contrastive_model.best"))
            # output embedding
            # entities.dict stores the id of the embedding
            contrastive_embedding = model.get_embedding(input_data_feature)
            #contrastive_embedding = F.normalize(contrastive_embedding, dim=1)
            contrastive_embedding = contrastive_embedding.detach().cpu().numpy()
            np.save(
                os.path.join(args.state_dir, args.task + '_contrastive_embedding'), 
                contrastive_embedding
            )

    # should save model
    # the new unified embedding is the output of h_i, h_j
    