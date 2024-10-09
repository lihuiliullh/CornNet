
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

EPSILON = float(np.finfo(float).eps)

def safe_log(x):
    return torch.log(x + EPSILON)

def entropy(p):
    return torch.sum(-p * safe_log(p), 1)




class ChoseHeadPolicyNetwork(nn.Module):
    def __init__(self, config, question_dim, node_dim, kg):
        super().__init__()
        self.input_dim = question_dim
        self.action_dim = node_dim
            
        self.ff_dropout_rate = 0.1
        self.kg = kg
        self.define_modules()
        self.config = config
    
    def define_modules(self):
        self.W1 = nn.Linear(self.input_dim, self.action_dim)
        self.W2 = nn.Linear(self.action_dim, self.action_dim)
        self.W1Dropout = nn.Dropout(p=self.ff_dropout_rate)
        self.W2Dropout = nn.Dropout(p=self.ff_dropout_rate)

        self.C1 = nn.Linear(self.action_dim, self.action_dim)
        self.C2 = nn.Linear(self.action_dim, 2)
        self.C1Dropout = nn.Dropout(p=self.ff_dropout_rate)
        self.C2Dropout = nn.Dropout(p=self.ff_dropout_rate)
    
    # each action is a node: head or answer of last step
    def get_action_embedding(self, node, kg):
        node_embedding = kg.get_entity_embeddings(node)
        return node_embedding
    
    def policy_nn_fun(self, question_emb, action_space, kg):
        # input X2 is a N x dim matrix
        # r_space is a N x action_num matrix, so is e_space and action_mask
        # change this to classifier
        X = question_emb
        X = self.C1(X)
        X = F.relu(X)
        X = self.C1Dropout(X)
        X = self.C2(X)
        X2 = self.C2Dropout(X) # embedding of history

        action_dist_without_softmax = X2
        action_dist = F.softmax(action_dist_without_softmax, dim=-1)
        # action_dist = ops.weighted_softmax(torch.squeeze(A @ torch.unsqueeze(X2, 2), 2), action_mask)
        return action_dist, entropy(action_dist), action_dist_without_softmax

    def transit(self, head, last_node, question_embedding, kg):
        X = question_embedding
        # MLP
        # policy network
        X = self.W1(X)
        X = F.relu(X)
        X = self.W1Dropout(X)
        X = self.W2(X)
        X2 = self.W2Dropout(X) # embedding of history

        action_space = torch.cat([head.unsqueeze(1), last_node.unsqueeze(1)], dim=-1)
        action_dist, entropy, action_dist_without_softmax = self.policy_nn_fun(X2, action_space, kg)
        
        return action_dist, entropy, action_space, action_dist_without_softmax