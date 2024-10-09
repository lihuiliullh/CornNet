import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torch.nn.init import xavier_normal_
from transformers import *
import random
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from RL.policy_network import *
from RL.chose_head_angent import *
from RL.environment import *
from RL.beam_search import *
from qa_vae import *
from transformer import TransformerEncoder


"""
This model is a pretrained model, it is pretrained to choose the current topic entity according to
the conversation history and current question.
According to the experiment, this model will choose the global topic entity as the current topic entity
most of the time.
"""

class MulticlassClassification(nn.Module):
    def __init__(self, input_dim, num_class):
        super(MulticlassClassification, self).__init__()

        #self.layer_1 = nn.Linear(num_feature, num_feature)
        self.layer_2 = nn.Linear(input_dim, int(input_dim * 1.2))
        self.layer_3 = nn.Linear(int(input_dim * 1.2), int(num_class * 0.8))
        self.layer_out = nn.Linear(int(num_class * 0.8), num_class)

        nn.init.xavier_uniform(self.layer_2.weight)
        nn.init.xavier_uniform(self.layer_3.weight)
        nn.init.xavier_uniform(self.layer_out.weight)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(input_dim)
        self.batchnorm2 = nn.BatchNorm1d(int(input_dim * 1.2))
        self.batchnorm3 = nn.BatchNorm1d(int(num_class * 0.8))
        self.m = nn.Softmax()

    def forward(self, x):
        #x = self.layer_1(x)
        #x = self.batchnorm1(x)
        #x = self.relu(x)
        #x = x.squeeze(1)

        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_out(x)
        x = self.m(x)

        #x = x.unsqueeze(1)

        return x



class RelationExtractor(nn.Module):

    def __init__(self, config, embedding_dim, num_entities, relation_emb, pretrained_embeddings, freeze, device, 
    entdrop=0.0, reldrop=0.0, scoredrop=0.0, l3_reg=0.0, ls=0.0, do_batch_norm=True):
        super(RelationExtractor, self).__init__()
        self.device = device
        self.freeze = freeze
        self.label_smoothing = ls
        self.l3_reg = l3_reg
        self.do_batch_norm = do_batch_norm
        if not self.do_batch_norm:
            print('Not doing batch norm')
        self.roberta_pretrained_weights = 'roberta-base'
        self.roberta_model = RobertaModel.from_pretrained(self.roberta_pretrained_weights)
        for param in self.roberta_model.parameters():
            param.requires_grad = True
    
        multiplier = 2
        self.getScores = self.ComplEx
        
        self.hidden_dim = 768
        
        self.num_entities = num_entities
        self.loss = self.kge_loss

        # best: all dropout 0
        self.rel_dropout = torch.nn.Dropout(reldrop)
        self.ent_dropout = torch.nn.Dropout(entdrop)
        self.score_dropout = torch.nn.Dropout(scoredrop)
        self.fcnn_dropout = torch.nn.Dropout(0.1)

        # self.pretrained_embeddings = pretrained_embeddings
        # random.shuffle(pretrained_embeddings)
        # print(pretrained_embeddings[0])
        print('Frozen:', self.freeze)
        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=self.freeze).to(self.device)
        # self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(pretrained_embeddings), freeze=self.freeze)
        print(self.embedding.weight.shape)

        
        self.lstm_layer_size = 2
        self.lstm = torch.nn.LSTM(self.hidden_dim, self.hidden_dim, self.lstm_layer_size, batch_first=True).to(self.device)

        relation_dim = self.embedding.weight.shape[1]
        self.relation_dim = relation_dim

        self.mid1 = 512
        self.mid2 = 512
        self.mid3 = 512
        self.mid4 = 512

        self.hidden2rel = nn.Linear(self.hidden_dim, self.relation_dim)
        self.hidden2rel_base = nn.Linear(self.mid2, self.relation_dim)

        self.bn0 = torch.nn.BatchNorm1d(multiplier)
        self.bn2 = torch.nn.BatchNorm1d(multiplier)

        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)        
        self._klloss = torch.nn.KLDivLoss(reduction='sum')

        

        self.relation_emb = nn.Embedding.from_pretrained(relation_emb, freeze=False).to(self.device)
        
        self.kg_environment = KGEnvironment(config, self.embedding, self.relation_emb)
        self.triple_emb = nn.Embedding(self.kg_environment.triple_num, self.relation_dim, max_norm=True).to(self.device)
        self.kg_environment.set_triple_embed(self.triple_emb)

        self.policy_network = PolicyNetwork(config, self.relation_dim, self.kg_environment)
        self.use_action_space_bucketing = True
        self.baseline = 'avg_reward'
        self.config = config
        self.rollout_num = config.rollout_num

        self.transformer = TransformerEncoder(self.hidden_dim, self.hidden_dim, 0.01, 0.01, 16, 4)
        # relation_dim = entity_dim
        self.chose_head_plicy = ChoseHeadPolicyNetwork(config, self.hidden_dim, self.relation_dim, self.kg_environment)

        # question_dim, latent_dim, relation_num, pretrained_relation_embeddings, device
        # relation_number add 1 for dummy relation
        #self.qa_vae_model = QaVAE(self.hidden_dim, self.relation_dim, self.relation_emb.weight.shape[0] + 1, self.relation_emb, self.device)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.head_chose_network = MulticlassClassification(self.relation_dim, 2)
        for name, param in self.head_chose_network.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)
        
        self.head_chose_loss = torch.nn.BCELoss(reduction='sum')

    def set_bn_eval(self):
        self.bn0.eval()
        self.bn2.eval()

    def kge_loss(self, scores, targets):
        # loss = torch.mean(scores*targets)
        return self._klloss(
            F.log_softmax(scores, dim=1), F.normalize(targets.float(), p=1, dim=1)
        )

    def applyNonLinear(self, outputs):
        outputs = self.hidden2rel(outputs)
        # outputs = self.hidden2rel_base(outputs)
        return outputs

    def ComplEx(self, head, relation):
        head = torch.stack(list(torch.chunk(head, 2, dim=1)), dim=1)
        if self.do_batch_norm:
            head = self.bn0(head)

        head = self.ent_dropout(head)
        relation = self.rel_dropout(relation)
        head = head.permute(1, 0, 2)
        re_head = head[0]
        im_head = head[1]

        re_relation, im_relation = torch.chunk(relation, 2, dim=1)
        re_tail, im_tail = torch.chunk(self.embedding.weight, 2, dim =1)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation

        score = torch.stack([re_score, im_score], dim=1)
        if self.do_batch_norm:
            score = self.bn2(score)

        score = self.score_dropout(score)
        score = score.permute(1, 0, 2)

        re_score = score[0]
        im_score = score[1]
        score = torch.mm(re_score, re_tail.transpose(1,0)) + torch.mm(im_score, im_tail.transpose(1,0))
        # pred = torch.sigmoid(score)
        pred = score
        return pred

    
    def getQuestionEmbedding(self, question_tokenized, attention_mask):
        roberta_last_hidden_states = self.roberta_model(question_tokenized, attention_mask=attention_mask)[0]
        states = roberta_last_hidden_states.transpose(1,0)
        cls_embedding = states[0]
        question_embedding = cls_embedding
        # question_embedding = torch.mean(roberta_last_hidden_states, dim=1)
        return question_embedding

    def sample_action(self, db_outcomes, inv_offset):
        def batch_lookup(M, idx, vector_output=True):
            """
            Perform batch lookup on matrix M using indices idx.
            :param M: (Variable) [batch_size, seq_len] Each row of M is an independent population.
            :param idx: (Variable) [batch_size, sample_size] Each row of idx is a list of sample indices.
            :param vector_output: If set, return a 1-D vector when sample size is 1.
            :return samples: [batch_size, sample_size] samples[i, j] = M[idx[i, j]]
            """
            batch_size, w = M.size()
            batch_size2, sample_size = idx.size()
            assert(batch_size == batch_size2)

            if sample_size == 1 and vector_output:
                samples = torch.gather(M, 1, idx).view(-1)
            else:
                samples = torch.gather(M, 1, idx)
            return samples

        # should be here, get top 50, then build embedding according to e, r, triple_id
        # use contrastive learning
        def sample(action_space, action_dist):
            ((r_space, e_space), action_mask, triple_id) = action_space
            idx = torch.multinomial(action_dist, 1, replacement=True)
            next_e = batch_lookup(e_space, idx)
            action_prob = batch_lookup(action_dist, idx)
            return next_e, action_prob


        if inv_offset is not None:
            # get the large one
            next_r_list = []
            next_e_list = []
            action_dist_list = []
            action_prob_list = []
            for action_space, action_dist in db_outcomes:
                # should restore the action
                # e[id1, id2, id3, useless, useless, ...], mask[1, 1, 1, 0, 0, 0,...], dist[v1, v2, v3, useless....]
                # matrix[0, .., v, ...v, ..]
                # I don't know how to solve the above
                # so I follow to sample one answer, if the correct probability is large, then the result should be correct
                next_e, next_prob = sample(action_space, action_dist)
                next_e_list.append(next_e)
                action_prob_list.append(next_prob)
            next_e = torch.cat(next_e_list, dim=0)[inv_offset]
            action_prob = torch.cat(action_prob_list, dim=0)[inv_offset]
        else:
            next_e, action_prob = sample(db_outcomes[0][0], db_outcomes[0][1])
        
        return next_e, action_prob

    def tile_along_beam(self, v, beam_size, dim=0):
        """
        Tile a tensor along a specified dimension for the specified beam size.
        :param v: Input tensor.
        :param beam_size: Beam size.
        """
        if dim == -1:
            dim = len(v.size()) - 1
        v = v.unsqueeze(dim + 1)
        v = torch.cat([v] * beam_size, dim=dim+1)
        new_size = []
        for i, d in enumerate(v.size()):
            if i == dim + 1:
                new_size[-1] *= d
            else:
                new_size.append(d)
        return v.view(new_size)
    
    def contrastive_loss(self, db_outcomes, inv_offset):
        loss = 0
        if inv_offset is not None:
            for action_space, action_dist in db_outcomes:
                ((r_space, e_space), action_mask, triple_id) = action_space
                # get top k
                val, idx = torch.topk(action_dist, 20, dim=1)

                r_top = torch.gather(r_space, 1, idx)
                e_top = torch.gather(e_space, 1, idx)
                triple_top = torch.gather(triple_id, 1, idx)

                #r_top = r_top.view(-1, 1)
                #e_top = e_top.view(-1, 1)
                #triple_top = triple_top.view(-1, 1)
                action_emb = self.policy_network.get_action_embedding((r_top, e_top, triple_top), self.kg_environment)
                loss += self.policy_network.cal_contrastive_loss(action_emb)
        else:
            action_space, action_dist = db_outcomes
            ((r_space, e_space), action_mask, triple_id) = action_space
            # get top k
            val, idx = torch.topk(action_dist, 20, dim=1)

            r_top = torch.zeros(idx.shape)
            e_top = torch.zeros(idx.shape)
            triple_top = torch.zeros(idx.shape)

            r_top = torch.scatter(r_top, 1, idx, r_space)
            e_top = torch.scatter(e_top, 1, idx, e_space)
            triple_top = torch.scatter(triple_top, 1, idx, triple_id)
            r_top = r_top.view(-1, 1)
            e_top = e_top.view(-1, 1)
            triple_top = triple_top.view(-1, 1)
            action_emb = self.policy_network.get_action_embedding((r_top, e_top, triple_top), self.kg_environment)
            loss += self.policy_network.cal_contrastive_loss(action_emb)
        
        return loss


    def operation(self, head, question_embedding, answer):
        db_outcomes, inv_offset, policy_entropy, contrastive_loss_sum = self.policy_network.transit(head, question_embedding, 
            self.kg_environment, use_action_space_bucketing=self.use_action_space_bucketing)
        sample_e, sample_prob = self.sample_action(db_outcomes, inv_offset)

        #contrastive_loss_sum = self.contrastive_loss(db_outcomes, inv_offset)
        contrastive_loss_sum = 0

        log_action_prob = safe_log(sample_prob)
        final_reward = self.reward_fun(sample_e, answer)
        #final_reward = self.stablize_reward(final_reward)

        # compute policy gradient
        pg_loss, pt_loss = 0, 0
        pg_loss = -final_reward * log_action_prob # model loss
        pt_loss = -final_reward * torch.exp(log_action_prob) # print loss

        # Entropy regularization
        action_entropy = policy_entropy
        entropy = action_entropy
        #entropy = torch.cat([x.unsqueeze(1) for x in action_entropy], dim=1).mean(dim=1)
        pg_loss = (pg_loss - entropy * self.config.beta).mean()
        pt_loss = (pt_loss - entropy * self.config.beta).mean()

        return pg_loss, pt_loss, contrastive_loss_sum, sample_e

    def head_choose(self, question_embedding):
        
        predict = self.head_chose_network(question_embedding)
        return predict

    def forward(self, p_head, question_tokenized1, attention_mask1, p_tail1,
        question_tokenized2, attention_mask2, p_tail2,
        question_tokenized3, attention_mask3, p_tail3,
        question_tokenized4, attention_mask4, p_tail4,
        question_tokenized5, attention_mask5, p_tail5,
        startpoint1, startpoint2, startpoint3, startpoint4, startpoint5
        ):

        head = p_head

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

        # predict 0 or 1 is enough 


        # choose head of the results of last step here
        pred1 = self.head_choose(rel_embedding1)
        pred2 = self.head_choose(rel_embedding2)
        pred3 = self.head_choose(rel_embedding3)
        pred4 = self.head_choose(rel_embedding4)
        pred5 = self.head_choose(rel_embedding5)
        
        # QA loss
        path_ground_truth1 = F.one_hot(startpoint1, num_classes=2)
        path_ground_truth2 = F.one_hot(startpoint2, num_classes=2)
        path_ground_truth3 = F.one_hot(startpoint3, num_classes=2)
        path_ground_truth4 = F.one_hot(startpoint4, num_classes=2)
        path_ground_truth5 = F.one_hot(startpoint5, num_classes=2)

        path_loss1 = self.head_chose_loss(pred1, path_ground_truth1.float())
        path_loss2 = self.head_chose_loss(pred2, path_ground_truth2.float())
        path_loss3 = self.head_chose_loss(pred3, path_ground_truth3.float())
        path_loss4 = self.head_chose_loss(pred4, path_ground_truth4.float())
        path_loss5 = self.head_chose_loss(pred5, path_ground_truth5.float())

        loss = path_loss1 + path_loss2 + path_loss3 + path_loss4 + path_loss5
        loss = loss.to(self.device)
        return loss
    

    def get_score_ranked(self, head, question_tokenized1, attention_mask1,
        question_tokenized2, attention_mask2,
        question_tokenized3, attention_mask3,
        question_tokenized4, attention_mask4,
        question_tokenized5, attention_mask5,
        startpoint1, startpoint2, startpoint3, startpoint4, startpoint5):

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


        pred1 = self.head_choose(rel_embedding1)
        pred2 = self.head_choose(rel_embedding2)
        pred3 = self.head_choose(rel_embedding3)
        pred4 = self.head_choose(rel_embedding4)
        pred5 = self.head_choose(rel_embedding5)
        
        pred_top1 = pred1.argmax(1)
        pred_top2 = pred2.argmax(1)
        pred_top3 = pred3.argmax(1)
        pred_top4 = pred4.argmax(1)
        pred_top5 = pred5.argmax(1)

        def calculate_error(list1, list2):
            error = 0
            for i in range(list1.shape[0]):
                a = list1[i]
                b = list2[i]
                if b == 0:
                    xxx = 0
                    #error += 1
                if b == 1:
                    error += 1
                #if a == 0 and b == 1:
                #    error += 1
            return error
        
        
        # a1 = (startpoint1 == pred_top1)
        # a2 = (startpoint2 == pred_top2)
        # a3 = (startpoint3 == pred_top3)
        # a4 = (startpoint4 == pred_top4)
        # a5 = (startpoint5 == pred_top5)

        # b1 = pred_top1.shape[0] - torch.sum(a1)
        # b2 = pred_top2.shape[0] - torch.sum(a2)
        # b3 = pred_top3.shape[0] - torch.sum(a3)
        # b4 = pred_top4.shape[0] - torch.sum(a4)
        # b5 = pred_top5.shape[0] - torch.sum(a5)

        b1 = calculate_error(startpoint1.cpu().detach().numpy(), pred_top1.cpu().detach().numpy())
        b2 = calculate_error(startpoint2.cpu().detach().numpy(), pred_top2.cpu().detach().numpy())
        b3 = calculate_error(startpoint3.cpu().detach().numpy(), pred_top3.cpu().detach().numpy())
        b4 = calculate_error(startpoint4.cpu().detach().numpy(), pred_top4.cpu().detach().numpy())
        b5 = calculate_error(startpoint5.cpu().detach().numpy(), pred_top5.cpu().detach().numpy())

        # top2 = torch.topk(scores, k=2, largest=True, sorted=True)
        # return top2
        return b1 + b2 + b3 + b4 + b5

