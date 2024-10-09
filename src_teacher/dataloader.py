import torch
import random
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import os
import unicodedata
import re
import time
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from tqdm.utils import RE_ANSI
from transformers import *

# KG and NLP use two different emb modules
# input data should be
# [topic_entity, Q1, answer1, Q2, answer2, ...]
# entity2idx for KG will map topic_entity -> ID, answer1 -> ID, ...
# tokenizer is a pretrained bert
class DatasetConversation(Dataset):
    def __init__(self, data, entity2idx, relation2idx, node_rels_map, ht2relation):
        self.data = data
        self.entity2idx = entity2idx
        self.tokenizer_class = RobertaTokenizer
        self.pretrained_weights = 'roberta-base'
        self.tokenizer = self.tokenizer_class.from_pretrained(self.pretrained_weights)
        self.pad_sequence_max_len = 64
        self.node_rels_map = node_rels_map
        self.relation2idx = relation2idx
        self.ht2relation = ht2relation

    def __len__(self):
        return len(self.data)
    
    def pad_sequence(self, arr, max_len=128):
        num_to_add = max_len - len(arr)
        if num_to_add < 0:
            return arr[0:max_len]
        for _ in range(num_to_add):
            arr.append('<pad>')
        return arr

    def toOneHot(self, indices):
        indices = torch.LongTensor(indices)
        batch_size = len(indices)
        vec_len = len(self.entity2idx)
        one_hot = torch.FloatTensor(vec_len)
        one_hot.zero_()
        # one_hot = -torch.ones(vec_len, dtype=torch.float32)
        one_hot.scatter_(0, indices, 1)
        return one_hot
    
    def relToOneHot2(self, indices):
        # sample 4 indices here
        num_rel = len(indices)
        indices = torch.LongTensor(indices)
        vec_len = len(self.relation2idx) + 1 # plus one for dummy relation
        one_hot = torch.FloatTensor(vec_len)
        one_hot.zero_()
        # one_hot = -torch.ones(vec_len, dtype=torch.float32)
        one_hot.scatter_(0, indices, 1)
        return one_hot
    
    # only sample one relation
    def sampleRelation(self, indices):
        # sample 4 indices here
        sample_size = 1
        if len(indices) >= sample_size:
            indices = indices[0:sample_size]
            sampled_neighbors = indices
        else:
            lef_len = sample_size - len(indices)
            tmp = np.random.choice(list(indices), size=lef_len, replace=len(indices) < sample_size)
            indices.extend(tmp)
            sampled_neighbors = indices
        
        return torch.LongTensor(sampled_neighbors)

    def get_relation_of_ht(self, head, tail):
        rel_ids = []
        # if no relations between two nodes
        # use dummy relation
        key = (head, tail)
        rels = []
        not_exist = False
        if key in self.ht2relation:
            rels = self.ht2relation[key]
        elif (tail, head) in self.ht2relation:
            rels = self.ht2relation[(tail, head)]
        else:
            not_exist = True

        if not not_exist:
            for rel_name in rels:
                rel_name = rel_name.strip()
                if rel_name in self.relation2idx:
                    rel_ids.append(self.relation2idx[rel_name])
                else:
                    error = 0
        else:
            rel_ids.append(len(self.relation2idx))
        rel_score = self.sampleRelation(rel_ids)
        return rel_score

    def __getitem__(self, index):
        data_point = self.data[index]
        topic_node = data_point[0]
        question_text1 = data_point[1]
        answer_text1 = data_point[2]
        question_text2 = data_point[3]
        answer_text2 = data_point[4]
        question_text3 = data_point[5]
        answer_text3 = data_point[6]
        question_text4 = data_point[7]
        answer_text4 = data_point[8]
        question_text5 = data_point[9]
        answer_text5 = data_point[10]

        question_tokenized1_list = []
        attention_mask1_list = []
        for q_txt in question_text1:
            question_tokenized1, attention_mask1 = self.tokenize_question(q_txt)
            question_tokenized1_list.append(question_tokenized1)
            attention_mask1_list.append(attention_mask1)

        question_tokenized2_list = []
        attention_mask2_list = []
        for q_txt in question_text2:
            question_tokenized2, attention_mask2 = self.tokenize_question(q_txt)
            question_tokenized2_list.append(question_tokenized2)
            attention_mask2_list.append(attention_mask2)

        question_tokenized3_list = []
        attention_mask3_list = []
        for q_txt in question_text3:
            question_tokenized3, attention_mask3 = self.tokenize_question(q_txt)
            question_tokenized3_list.append(question_tokenized3)
            attention_mask3_list.append(attention_mask3)
        
        question_tokenized4_list = []
        attention_mask4_list = []
        for q_txt in question_text4:
            question_tokenized4, attention_mask4 = self.tokenize_question(q_txt)
            question_tokenized4_list.append(question_tokenized4)
            attention_mask4_list.append(attention_mask4)

        question_tokenized5_list = []
        attention_mask5_list = []
        for q_txt in question_text5:
            question_tokenized5, attention_mask5 = self.tokenize_question(q_txt)
            question_tokenized5_list.append(question_tokenized5)
            attention_mask5_list.append(attention_mask5)

        topic_node_id = self.entity2idx[topic_node.strip()]
        answer_text1_id = self.entity2idx[answer_text1.strip()]
        answer_text2_id = self.entity2idx[answer_text2.strip()]
        answer_text3_id = self.entity2idx[answer_text3.strip()]
        answer_text4_id = self.entity2idx[answer_text4.strip()]
        answer_text5_id = self.entity2idx[answer_text5.strip()]

        # answer_text1_id = self.toOneHot([answer_text1_id])
        # answer_text2_id = self.toOneHot([answer_text2_id])
        # answer_text3_id = self.toOneHot([answer_text3_id])
        # answer_text4_id = self.toOneHot([answer_text4_id])
        # answer_text5_id = self.toOneHot([answer_text5_id])
        # also return the question
        # also return the multi-hot vector of neighbor
        # should return the relation answer as well

        question_tokenized1_list = torch.stack(question_tokenized1_list, dim=0)
        attention_mask1_list = torch.stack(attention_mask1_list, dim=0)
        try:
            question_tokenized2_list = torch.stack(question_tokenized2_list, dim=0)
        except:
            aa = 0
        attention_mask2_list = torch.stack(attention_mask2_list, dim=0)
        question_tokenized3_list = torch.stack(question_tokenized3_list, dim=0)
        attention_mask3_list = torch.stack(attention_mask3_list, dim=0)
        question_tokenized4_list = torch.stack(question_tokenized4_list, dim=0)
        attention_mask4_list = torch.stack(attention_mask4_list, dim=0)
        question_tokenized5_list = torch.stack(question_tokenized5_list, dim=0)
        attention_mask5_list = torch.stack(attention_mask5_list, dim=0)
        
        return topic_node_id, question_tokenized1_list, attention_mask1_list, answer_text1_id, question_tokenized2_list, attention_mask2_list, answer_text2_id, \
            question_tokenized3_list, attention_mask3_list, answer_text3_id, question_tokenized4_list, attention_mask4_list, answer_text4_id, \
            question_tokenized5_list, attention_mask5_list, answer_text5_id, question_text1[0], question_text2[0], question_text3[0], question_text4[0], question_text5[0], \
            data_point[11], data_point[12], data_point[13], data_point[14], data_point[15]

    def tokenize_question(self, question):
        question = "<s> " + question + " </s>"
        question_tokenized = self.tokenizer.tokenize(question)
        question_tokenized = self.pad_sequence(question_tokenized, self.pad_sequence_max_len)
        question_tokenized = torch.tensor(self.tokenizer.encode(question_tokenized, add_special_tokens=False))
        attention_mask = []
        for q in question_tokenized:
            # 1 means padding token
            if q == 1:
                attention_mask.append(0)
            else:
                attention_mask.append(1)
        return question_tokenized, torch.tensor(attention_mask, dtype=torch.long)

# class DatasetConversationTest(Dataset):
#     def __init__(self, data, entity2idx):
#         self.data = data
#         self.entity2idx = entity2idx
#         self.tokenizer_class = RobertaTokenizer
#         self.pretrained_weights = 'roberta-base'
#         self.tokenizer = self.tokenizer_class.from_pretrained(self.pretrained_weights)
#         self.pad_sequence_max_len = 64

#     def __len__(self):
#         return len(self.data)
    
#     def pad_sequence(self, arr, max_len=128):
#         num_to_add = max_len - len(arr)
#         for _ in range(num_to_add):
#             arr.append('<pad>')
#         return arr

#     def __getitem__(self, index):
#         data_point = self.data[index]
#         topic_node = data_point[0]
#         question_text1 = data_point[1]
#         answer_text1 = data_point[2]
#         question_text2 = data_point[3]
#         answer_text2 = data_point[4]
#         question_text3 = data_point[5]
#         answer_text3 = data_point[6]
#         question_text4 = data_point[7]
#         answer_text4 = data_point[8]
#         question_text5 = data_point[9]
#         answer_text5 = data_point[10]

#         question_tokenized1, attention_mask1 = self.tokenize_question(question_text1)
#         question_tokenized2, attention_mask2 = self.tokenize_question(question_text2)
#         question_tokenized3, attention_mask3 = self.tokenize_question(question_text3)
#         question_tokenized4, attention_mask4 = self.tokenize_question(question_text4)
#         question_tokenized5, attention_mask5 = self.tokenize_question(question_text5)

#         topic_node_id = self.entity2idx[topic_node.strip()]
#         answer_text1_id = self.entity2idx[answer_text1.strip()]
#         answer_text2_id = self.entity2idx[answer_text2.strip()]
#         answer_text3_id = self.entity2idx[answer_text3.strip()]
#         answer_text4_id = self.entity2idx[answer_text4.strip()]
#         answer_text5_id = self.entity2idx[answer_text5.strip()]

#         return topic_node_id, question_tokenized1, attention_mask1, answer_text1_id, question_tokenized2, attention_mask2, answer_text2_id, \
#             question_tokenized3, attention_mask3, answer_text3_id, question_tokenized4, attention_mask4, answer_text4_id, \
#             question_tokenized5, attention_mask5, answer_text5_id, question_text1, question_text2, question_text3, question_text4, question_text5

#     def tokenize_question(self, question):
#         question = "<s> " + question + " </s>"
#         question_tokenized = self.tokenizer.tokenize(question)
#         question_tokenized = self.pad_sequence(question_tokenized, self.pad_sequence_max_len)
#         question_tokenized = torch.tensor(self.tokenizer.encode(question_tokenized, add_special_tokens=False))
#         attention_mask = []
#         for q in question_tokenized:
#             # 1 means padding token
#             if q == 1:
#                 attention_mask.append(0)
#             else:
#                 attention_mask.append(1)
#         return question_tokenized, torch.tensor(attention_mask, dtype=torch.long)