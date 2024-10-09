


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .contrastive_learning import *

HUGE_INT = 1e31
EPSILON = float(np.finfo(float).eps)

def safe_log(x):
    return torch.log(x + EPSILON)

def entropy(p):
    return torch.sum(-p * safe_log(p), 1)

class PolicyNetwork(nn.Module):
    def __init__(self, config, input_dim, kg):
        super().__init__()
        self.relation_only = False
        self.input_dim = input_dim

        if not self.relation_only:
            self.action_dim = 3 * input_dim
        else:
            self.action_dim = input_dim
            
        self.ff_dropout_rate = 0.1
        self.kg = kg
        self.define_modules()
        self.config = config
        #self.simCLR = SimCLR(self.action_dim, 128, config)
    
    def define_modules(self):
        self.W1 = nn.Linear(self.input_dim, self.action_dim)
        self.W2 = nn.Linear(self.action_dim, self.action_dim)
        self.W1Dropout = nn.Dropout(p=self.ff_dropout_rate)
        self.W2Dropout = nn.Dropout(p=self.ff_dropout_rate)
        
    def get_action_embedding(self, action, kg):
        r, e, triple_id = action
        relation_embedding = kg.get_relation_embeddings(r)
        #relation_context_embedding = relation_context_embedding.unsqueeze(0)
        if self.relation_only:
            action_embedding = relation_embedding
        else:
            entity_embedding = kg.get_entity_embeddings(e)
            triple_embedding = kg.get_triple_embeddings(triple_id)
            action_embedding = torch.cat([relation_embedding, entity_embedding, triple_embedding], dim=-1)
            a = 0
        return action_embedding
    
    def cal_contrastive_loss(self, A):
        contrastive_loss = self.simCLR(A)
        return contrastive_loss
    
    def policy_nn_fun(self, X2, action_space, kg):
        # input X2 is a N x dim matrix
        # r_space is a N x action_num matrix, so is e_space and action_mask
        (r_space, e_space), action_mask, triple_id = action_space
        A = self.get_action_embedding((r_space, e_space, triple_id), kg)
        # a easy way is to add contrastive learning here.
        # if self.config.training:
        #     contrastive_loss = self.simCLR(A)
        # else:
        #     contrastive_loss = 0
        action_dist = F.softmax(
            torch.squeeze(A @ torch.unsqueeze(X2, 2), 2) - (1 - action_mask) * HUGE_INT, dim=-1)
        # action_dist = ops.weighted_softmax(torch.squeeze(A @ torch.unsqueeze(X2, 2), 2), action_mask)
        return action_dist, entropy(action_dist), 0

    # choose action
    def transit(self, head, obs, kg, use_action_space_bucketing=True):
        # obs should be the embedding of question
        # in KG setting: obs is (e_s, q, all_other_information_see_previous)
        question_embedding = obs
        X = question_embedding
        # MLP
        # policy network
        X = self.W1(X)
        X = F.relu(X)
        X = self.W1Dropout(X)
        X = self.W2(X)
        X2 = self.W2Dropout(X) # embedding of history

        contrastive_loss_sum = 0
        if use_action_space_bucketing:
            db_outcomes = []
            entropy_list = []
            references = []
            db_action_spaces, db_references_orginIndx = kg.get_action_space_in_buckets(head)
            for action_space_b, index_in_batch in zip(db_action_spaces, db_references_orginIndx):
                X2_b = X2[index_in_batch, :]
                action_dist_b, entropy_b, contrastive_loss = self.policy_nn_fun(X2_b, action_space_b, self.kg)
                contrastive_loss_sum = contrastive_loss_sum + contrastive_loss
                references.extend(index_in_batch)
                db_outcomes.append((action_space_b, action_dist_b))
                entropy_list.append(entropy_b)
            # this one is really great
            inv_offset = [i for i, _ in sorted(enumerate(references), key=lambda x: x[1])]
            entropy = torch.cat(entropy_list, dim=0)[inv_offset]
        else:
            action_space = kg.get_action_space(head)
            action_dist, entropy, contrastive_loss = self.policy_nn_fun(X2, action_space)
            contrastive_loss_sum = contrastive_loss_sum + contrastive_loss
            db_outcomes = [(action_space, action_dist)]
            inv_offset = None
        
        return db_outcomes, inv_offset, entropy, contrastive_loss_sum


