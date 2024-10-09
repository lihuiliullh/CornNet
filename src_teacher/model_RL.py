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
from transformer import TransformerEncoder

"""
This one is the RL model describe in paper (the CornNet)
"""

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

    def choose_anchor_node(self, head, last_node, question_embedding, startnode=None):
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
        
        
        action_dist, entropy, action_space, action_dist_without_softmax = self.chose_head_plicy.transit(head, last_node, question_embedding, self.kg_environment)
        if startnode is not None:
            loss = self.cross_entropy_loss(action_dist_without_softmax, startnode)
            # sample one
            idx = torch.multinomial(action_dist, 1, replacement=True)
            next_e = batch_lookup(action_space, idx)
            startnode = startnode.unsqueeze(1)
            true_e = batch_lookup(action_space, startnode)
            return next_e, loss ,true_e
        else:
            next_e_prob, idx_ = torch.topk(action_dist, 1)
            next_e = batch_lookup(action_space, idx_)
            return next_e

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

        head = self.tile_along_beam(head, self.rollout_num)
        rel_embedding1 = self.tile_along_beam(rel_embedding1, self.rollout_num)
        rel_embedding2 = self.tile_along_beam(rel_embedding2, self.rollout_num)
        rel_embedding3 = self.tile_along_beam(rel_embedding3, self.rollout_num)
        rel_embedding4 = self.tile_along_beam(rel_embedding4, self.rollout_num)
        rel_embedding5 = self.tile_along_beam(rel_embedding5, self.rollout_num)
        p_tail1 = self.tile_along_beam(p_tail1, self.rollout_num)
        p_tail2 = self.tile_along_beam(p_tail2, self.rollout_num)
        p_tail3 = self.tile_along_beam(p_tail3, self.rollout_num)
        p_tail4 = self.tile_along_beam(p_tail4, self.rollout_num)
        p_tail5 = self.tile_along_beam(p_tail5, self.rollout_num)

        # choose head of the results of last step here
        pg_loss1, pt_loss1, contrastive_loss_sum1, answer_node1 = self.operation(head, rel_embedding1, p_tail1)

        # choose between head and answer node
        # input is head embedding, relation embedding answer embedding

        #lstm_rel_embedding2 = self.tile_along_beam(output2.squeeze(1), self.rollout_num)
        #startnode2 = self.tile_along_beam(startpoint2, self.rollout_num)
        #anchor_node, choose_loss2, true_e2 = self.choose_anchor_node(head, answer_node1, lstm_rel_embedding2, startnode2)
        pg_loss2, pt_loss2, contrastive_loss_sum2, answer_node2 = self.operation(head.long(), rel_embedding2, p_tail2)

        #lstm_rel_embedding3 = self.tile_along_beam(output3.squeeze(1), self.rollout_num)
        #startnode3 = self.tile_along_beam(startpoint3, self.rollout_num)
        #anchor_node, choose_loss3, true_e3 = self.choose_anchor_node(head, answer_node2, lstm_rel_embedding3, startnode3)
        pg_loss3, pt_loss3, contrastive_loss_sum3, answer_node3 = self.operation(head.long(), rel_embedding3, p_tail3)

        #lstm_rel_embedding4 = self.tile_along_beam(output4.squeeze(1), self.rollout_num)
        #startnode4 = self.tile_along_beam(startpoint4, self.rollout_num)
        #anchor_node, choose_loss4, true_e4 = self.choose_anchor_node(head, answer_node3, lstm_rel_embedding4, startnode4)
        pg_loss4, pt_loss4, contrastive_loss_sum4, answer_node4 = self.operation(head.long(), rel_embedding4, p_tail4)

        #lstm_rel_embedding5 = self.tile_along_beam(output5.squeeze(1), self.rollout_num)
        #startnode5 = self.tile_along_beam(startpoint5, self.rollout_num)
        #anchor_node, choose_loss5, true_e5 = self.choose_anchor_node(head, answer_node4, lstm_rel_embedding5, startnode5)
        pg_loss5, pt_loss5, contrastive_loss_sum5, answer_node5 = self.operation(head.long(), rel_embedding5, p_tail5)

        # QA loss
        loss = pg_loss1 + pg_loss2 + pg_loss3 + pg_loss4 + pg_loss5
        #loss = loss + (choose_loss2 + choose_loss3 + choose_loss4 + choose_loss5) * 10
        loss = loss.to(self.device)
        return loss
    

    def get_score_ranked(self, head, question_tokenized1, attention_mask1,
        question_tokenized2, attention_mask2,
        question_tokenized3, attention_mask3,
        question_tokenized4, attention_mask4,
        question_tokenized5, attention_mask5):

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

        # find answer
        # beam search
        # choose the nodes with the highest probability
        # nodes already sorted
        nodes1, log_action_prob1 = beam_search(self.policy_network, head, rel_embedding1, self.kg_environment)

        #lstm_rel_embedding2 = output2.squeeze(1)
        #anchor_node = self.choose_anchor_node(head, nodes1[:,0], lstm_rel_embedding2)
        nodes2, log_action_prob2 = beam_search(self.policy_network, head.long(), rel_embedding2, self.kg_environment)

        #lstm_rel_embedding3 = output3.squeeze(1)
        #anchor_node = self.choose_anchor_node(head, nodes2[:,0], lstm_rel_embedding3)
        nodes3, log_action_prob3 = beam_search(self.policy_network, head.long(), rel_embedding3, self.kg_environment)

        #lstm_rel_embedding4 = output4.squeeze(1)
        #anchor_node = self.choose_anchor_node(head, nodes3[:,0], lstm_rel_embedding4)
        nodes4, log_action_prob4 = beam_search(self.policy_network, head.long(), rel_embedding4, self.kg_environment)

        #lstm_rel_embedding5 = output5.squeeze(1)
        #anchor_node = self.choose_anchor_node(head, nodes4[:,0], lstm_rel_embedding5)
        nodes5, log_action_prob5 = beam_search(self.policy_network, head.long(), rel_embedding5, self.kg_environment)

        
        # top2 = torch.topk(scores, k=2, largest=True, sorted=True)
        # return top2
        return nodes1, nodes2, nodes3, nodes4, nodes5

    

    def reward_fun(self, pred, ground_truth):
        return (pred == ground_truth).float()
    
    def stablize_reward(self, r):
        r_2D = r.view(-1, 1)
        if self.baseline == 'avg_reward':
            stabled_r_2D = r_2D - r_2D.mean(dim=1, keepdim=True)
        elif self.baseline == 'avg_reward_normalized':
            stabled_r_2D = (r_2D - r_2D.mean(dim=1, keepdim=True)) / (r_2D.std(dim=1, keepdim=True) + ops.EPSILON)
        else:
            raise ValueError('Unrecognized baseline function: {}'.format(self.baseline))
        stabled_r = stabled_r_2D.view(-1)
        return stabled_r

        

