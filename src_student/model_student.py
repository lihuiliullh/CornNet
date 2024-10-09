import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils
from torch.autograd import Variable
from torch.nn.init import xavier_normal_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import *

from qa_vae import *
from RL.beam_search import *
from RL.environment import *
from RL.policy_network import *
from transformer import TransformerEncoder


class ReformulationImitator(nn.Module):

    def __init__(self, config, embedding_dim, num_entities, relation_emb, pretrained_embeddings, freeze, device, 
    entdrop=0.0, reldrop=0.0, scoredrop=0.0, l3_reg=0.0, ls=0.0, do_batch_norm=True):
        super(ReformulationImitator, self).__init__()
        self.device = device
        self.freeze = False
        self.label_smoothing = ls
        self.l3_reg = l3_reg
        self.do_batch_norm = do_batch_norm
        if not self.do_batch_norm:
            print('Not doing batch norm')
        self.roberta_pretrained_weights = 'roberta-base'
        self.roberta_model = RobertaModel.from_pretrained(self.roberta_pretrained_weights)
        for param in self.roberta_model.parameters():
            param.requires_grad = True
    
        self.hidden_dim = 768
        
        self.num_entities = num_entities

        self.lstm_layer_size = 2
        self.lstm = torch.nn.LSTM(self.hidden_dim, self.hidden_dim, self.lstm_layer_size, batch_first=True).to(self.device)

        relation_dim = pretrained_embeddings.shape[1]
        self.relation_dim = relation_dim

        self.hidden2rel = nn.Linear(self.hidden_dim, self.relation_dim)
        
        self.config = config
        
        self.transformer = TransformerEncoder(self.hidden_dim, self.hidden_dim, 0.01, 0.01, 16, 4)

    
    def applyNonLinear(self, outputs):
        outputs = self.hidden2rel(outputs)
        return outputs

    def getQuestionEmbedding(self, question_tokenized, attention_mask):
        roberta_last_hidden_states = self.roberta_model(question_tokenized, attention_mask=attention_mask)[0]
        states = roberta_last_hidden_states.transpose(1,0)
        cls_embedding = states[0]
        question_embedding = cls_embedding
        # question_embedding = torch.mean(roberta_last_hidden_states, dim=1)
        return question_embedding


    def forward(self, p_head, question_tokenized1, attention_mask1, p_tail1,
        question_tokenized2, attention_mask2, p_tail2,
        question_tokenized3, attention_mask3, p_tail3,
        question_tokenized4, attention_mask4, p_tail4,
        question_tokenized5, attention_mask5, p_tail5,
        rel_ans1, rel_ans2, rel_ans3, rel_ans4, rel_ans5
        ):
        question_shape = question_tokenized1.shape
        mask_shape = attention_mask1.shape

        question_tokenized1 = question_tokenized1.view(-1, question_shape[-1])
        attention_mask1 = attention_mask1.view(-1, mask_shape[-1])
        question_tokenized2 = question_tokenized2.view(-1, question_shape[-1])
        attention_mask2 = attention_mask2.view(-1, mask_shape[-1])
        question_tokenized3 = question_tokenized3.view(-1, question_shape[-1])
        attention_mask3 = attention_mask3.view(-1, mask_shape[-1])
        question_tokenized4 = question_tokenized4.view(-1, question_shape[-1])
        attention_mask4 = attention_mask4.view(-1, mask_shape[-1])
        question_tokenized5 = question_tokenized5.view(-1, question_shape[-1])
        attention_mask5 = attention_mask5.view(-1, mask_shape[-1])

        question_embedding1 = self.getQuestionEmbedding(question_tokenized1, attention_mask1)
        question_embedding2 = self.getQuestionEmbedding(question_tokenized2, attention_mask2)
        question_embedding3 = self.getQuestionEmbedding(question_tokenized3, attention_mask3)
        question_embedding4 = self.getQuestionEmbedding(question_tokenized4, attention_mask4)
        question_embedding5 = self.getQuestionEmbedding(question_tokenized5, attention_mask5)

        # use multi-head attention here
        question_shape = list(question_shape)
        question_shape[-1] = -1
        question_embedding1 = question_embedding1.view(question_shape)
        question_embedding2 = question_embedding2.view(question_shape)
        question_embedding3 = question_embedding3.view(question_shape)
        question_embedding4 = question_embedding4.view(question_shape)
        question_embedding5 = question_embedding5.view(question_shape)
        # use multi head attention here
        question_embedding1 = self.transformer(question_embedding1)
        question_embedding2 = self.transformer(question_embedding2)
        question_embedding3 = self.transformer(question_embedding3)
        question_embedding4 = self.transformer(question_embedding4)
        question_embedding5 = self.transformer(question_embedding5)

        question_embedding1 = question_embedding1[:,0,:]
        question_embedding2 = question_embedding2[:,0,:]
        question_embedding3 = question_embedding3[:,0,:]
        question_embedding4 = question_embedding4[:,0,:]
        question_embedding5 = question_embedding5[:,0,:]

        question_embedding1 = question_embedding1.unsqueeze(1)
        question_embedding2 = question_embedding2.unsqueeze(1)
        question_embedding3 = question_embedding3.unsqueeze(1)
        question_embedding4 = question_embedding4.unsqueeze(1)
        question_embedding5 = question_embedding5.unsqueeze(1)

        h_state = torch.zeros(self.lstm_layer_size, question_embedding1.size(0), self.hidden_dim, requires_grad=False).to(question_embedding1.device)
        c_state = torch.zeros(self.lstm_layer_size, question_embedding1.size(0), self.hidden_dim, requires_grad=False).to(question_embedding1.device)

        # output shoule be the same as h, check
        output1, (h1, c1) = self.lstm(question_embedding1, (h_state, c_state))
        output2, (h2, c2) = self.lstm(question_embedding2, (h1, c1))
        output3, (h3, c3) = self.lstm(question_embedding3, (h2, c2))
        output4, (h4, c4) = self.lstm(question_embedding4, (h3, c3))
        output5, (h5, c5) = self.lstm(question_embedding5, (h4, c4))


        rel_embedding1 = self.applyNonLinear(output1.squeeze(1))
        rel_embedding2 = self.applyNonLinear(output2.squeeze(1))
        rel_embedding3 = self.applyNonLinear(output3.squeeze(1))
        rel_embedding4 = self.applyNonLinear(output4.squeeze(1))
        rel_embedding5 = self.applyNonLinear(output5.squeeze(1))

        return rel_embedding1, rel_embedding2, rel_embedding3, rel_embedding4, rel_embedding5
    
        # only use question here, don't use reformulation. index 0 is the original question. 
        
        # question_tokenized1 = question_tokenized1[:,0,:]
        # question_tokenized2 = question_tokenized2[:,0,:]
        # question_tokenized3 = question_tokenized3[:,0,:]
        # question_tokenized4 = question_tokenized4[:,0,:]
        # question_tokenized5 = question_tokenized5[:,0,:]
        # attention_mask1 = attention_mask1[:,0,:]
        # attention_mask2 = attention_mask2[:,0,:]
        # attention_mask3 = attention_mask3[:,0,:]
        # attention_mask4 = attention_mask4[:,0,:]
        # attention_mask5 = attention_mask5[:,0,:]

        # question_embedding1 = self.getQuestionEmbedding(question_tokenized1, attention_mask1)
        # question_embedding2 = self.getQuestionEmbedding(question_tokenized2, attention_mask2)
        # question_embedding3 = self.getQuestionEmbedding(question_tokenized3, attention_mask3)
        # question_embedding4 = self.getQuestionEmbedding(question_tokenized4, attention_mask4)
        # question_embedding5 = self.getQuestionEmbedding(question_tokenized5, attention_mask5)

        # question_embedding1 = question_embedding1.unsqueeze(1)
        # question_embedding2 = question_embedding2.unsqueeze(1)
        # question_embedding3 = question_embedding3.unsqueeze(1)
        # question_embedding4 = question_embedding4.unsqueeze(1)
        # question_embedding5 = question_embedding5.unsqueeze(1)

        # h_state = torch.zeros(self.lstm_layer_size, question_embedding1.size(0), self.hidden_dim, requires_grad=False).to(question_embedding1.device)
        # c_state = torch.zeros(self.lstm_layer_size, question_embedding1.size(0), self.hidden_dim, requires_grad=False).to(question_embedding1.device)

        # # output shoule be the same as h, check
        # output1, (h1, c1) = self.lstm(question_embedding1, (h_state, c_state))
        # output2, (h2, c2) = self.lstm(question_embedding2, (h1, c1))
        # output3, (h3, c3) = self.lstm(question_embedding3, (h2, c2))
        # output4, (h4, c4) = self.lstm(question_embedding4, (h3, c3))
        # output5, (h5, c5) = self.lstm(question_embedding5, (h4, c4))


        # rel_embedding1 = self.applyNonLinear(output1.squeeze(1))
        # rel_embedding2 = self.applyNonLinear(output2.squeeze(1))
        # rel_embedding3 = self.applyNonLinear(output3.squeeze(1))
        # rel_embedding4 = self.applyNonLinear(output4.squeeze(1))
        # rel_embedding5 = self.applyNonLinear(output5.squeeze(1))

        # return rel_embedding1, rel_embedding2, rel_embedding3, rel_embedding4, rel_embedding5
