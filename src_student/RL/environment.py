
import collections
import os
import pickle

import torch
import torch.nn as nn

class KGEnvironment(nn.Module):

    def __init__(self, config, entity_embd, relation_embed):
        super().__init__()
        self.entity2id = config.entity2id
        self.id2entity = config.id2entity
        self.relation2id = config.relation2id
        self.id2relation = config.id2relation
        self.config = config
        self.device = config.device

        # this two will be initialized in vectorize_action_space
        self.action_space = None
        self.action_space_buckets = None

        self.bandwidth_maxActionNum = 300
        self.num_entities = config.num_entities
        self.entity_emb = entity_embd
        self.relation_emb = relation_embed
        
        self.adj_list = config.adj_list

        self.triple_num = 0
        self.vectorize_action_space()
        
    
    def set_triple_embed(self, triple_embed):
        self.triple_emb = triple_embed
    
    def get_relation_embeddings(self, r):
        return self.relation_emb(r.long())
    
    def get_entity_embeddings(self, e):
        return self.entity_emb(e.long())
    
    def get_triple_embeddings(self, triple_id):
        return self.triple_emb(triple_id.long())
    
    def vectorize_action_space(self):
        # should do something on pagerank KG

        # load page rank scores
        def load_page_rank_scores(input_path):
            pgrk_scores = collections.defaultdict(float)
            with open(input_path) as f:
                for line in f:
                    e, score = line.strip().split('\t')
                    e_id = self.entity2id[e.strip()]
                    score = float(score)
                    pgrk_scores[e_id] = score
            return pgrk_scores

        page_rank_scores = load_page_rank_scores(self.config.page_rank_file)

        # get a unique id for each triple
        def get_action_space(e1):
            action_space = []
            if e1 in self.adj_list:
                for r in self.adj_list[e1]:
                    targets = self.adj_list[e1][r]
                    for e2 in targets:
                        action_space.append((r, e2))
                # here may cause trouble
                if len(action_space) + 1 >= self.bandwidth_maxActionNum:
                    sorted_action_space = \
                        sorted(action_space, key=lambda x: page_rank_scores[x[1]], reverse=True)
                    action_space = sorted_action_space[:self.bandwidth_maxActionNum]
            return action_space
        
        # action_space_size is the max number of actions a node has
        def vectorize_action_space(action_space_list, action_space_size):
            bucket_size = len(action_space_list)
            r_space = torch.zeros(bucket_size, action_space_size, requires_grad=False).to(self.device)
            e_space = torch.zeros(bucket_size, action_space_size, requires_grad=False).to(self.device)
            triple_id_space = torch.zeros(bucket_size, action_space_size, requires_grad=False).to(self.device)
            action_mask = torch.zeros(bucket_size, action_space_size, requires_grad=False).to(self.device)
            for i, action_space in enumerate(action_space_list):
                for j, (r, e) in enumerate(action_space):
                    r_space[i, j] = r
                    e_space[i, j] = e
                    triple_id_space[i, j] = self.triple_num
                    self.triple_num = self.triple_num + 1
                    action_mask[i, j] = 1
            return (r_space, e_space), action_mask, triple_id_space

        if self.config.use_action_space_bucketing:
            self.action_space_buckets = {}
            action_space_buckets_discrete = collections.defaultdict(list)
            self.entity2bucketid = torch.zeros(self.num_entities, 2).long()
            for e1 in range(self.num_entities):
                # [(r, e2)]
                action_space = get_action_space(e1)
                key = int(len(action_space) / self.config.bucket_interval) + 1
                self.entity2bucketid[e1, 0] = key
                self.entity2bucketid[e1, 1] = len(action_space_buckets_discrete[key])
                # the size is the number of nodes
                action_space_buckets_discrete[key].append(action_space)
            for key in action_space_buckets_discrete:
                self.action_space_buckets[key] = vectorize_action_space(
                    action_space_buckets_discrete[key], key * self.config.bucket_interval)
        else:
            action_space_list = []
            max_num_actions = 0
            for e1 in range(self.num_entities):
                action_space = get_action_space(e1)
                action_space_list.append(action_space)
                if len(action_space) > max_num_actions:
                    max_num_actions = len(action_space)
            self.action_space = vectorize_action_space(action_space_list, max_num_actions)


    def get_action_space_in_buckets(self, head):
        """
        To compute the search operation in batch, we group the action spaces of different states
        (i.e. the set of outgoing edges of different nodes) into buckets based on their sizes to
        save the memory consumption of paddings.

        For example, in large knowledge graphs, certain nodes may have thousands of outgoing
        edges while a long tail of nodes only have a small amount of outgoing edges. If a batch
        contains a node with 1000 outgoing edges while the rest of the nodes have a maximum of
        5 outgoing edges, we need to pad the action spaces of all nodes to 1000, which consumes
        lots of memory.

        With the bucketing approach, each bucket is padded separately. In this case the node
        with 1000 outgoing edges will be in its own bucket and the rest of the nodes will suffer
        little from padding the action space to 5.

        Once we grouped the action spaces in buckets, the policy network computation is carried
        out for every bucket iteratively. Once all the computation is done, we concatenate the
        results of all buckets and restore their original order in the batch. The computation
        outside the policy network module is thus unaffected.

        :return db_action_spaces:
            [((r_space_b0, r_space_b0), action_mask_b0),
             ((r_space_b1, r_space_b1), action_mask_b1),
             ...
             ((r_space_bn, r_space_bn), action_mask_bn)]

            A list of action space tensor representations grouped in n buckets, s.t.
            r_space_b0.size(0) + r_space_b1.size(0) + ... + r_space_bn.size(0) = e.size(0)

        :return db_references:
            [l_batch_refs0, l_batch_refs1, ..., l_batch_refsn]
            l_batch_refsi stores the indices of the examples in bucket i in the current batch,
            which is used later to restore the output results to the original order.
        """
        # e_s: source. q: query. They are global
        # e_t: the entity visited at step t
        db_action_spaces, db_references_orginIndx = [], []


        entity2bucketid = self.entity2bucketid[head]
        key1 = entity2bucketid[:, 0]
        key2 = entity2bucketid[:, 1]
        bucketID_to_batch_idx = {}
        for i in range(len(head)):
            bucketID = int(key1[i])
            if not bucketID in bucketID_to_batch_idx:
                bucketID_to_batch_idx[bucketID] = []
            bucketID_to_batch_idx[bucketID].append(i)
        for bucketID in bucketID_to_batch_idx:
            action_space = self.action_space_buckets[bucketID]
            # l_batch_refs: ids of the examples in the current batch of examples
            # g_bucket_ids: ids of the examples in the corresponding KG action space bucket
            batch_idx = bucketID_to_batch_idx[bucketID]
            g_id_inside_bucket = key2[batch_idx].tolist()
            r_space_b = action_space[0][0][g_id_inside_bucket]
            e_space_b = action_space[0][1][g_id_inside_bucket]
            action_mask_b = action_space[1][g_id_inside_bucket]
            triple_id_b = action_space[2][g_id_inside_bucket]
            e_b = head[batch_idx]

            action_space_b = ((r_space_b, e_space_b), action_mask_b, triple_id_b)

            db_action_spaces.append(action_space_b)
            db_references_orginIndx.append(batch_idx)

        return db_action_spaces, db_references_orginIndx


    def get_action_space(self, e):
        r_space, e_space = self.action_space[0][0][e], self.action_space[0][1][e]
        action_mask = self.action_space[1][e]
        triple_id_space = self.action_space[2][e]
        action_space = ((r_space, e_space), action_mask, triple_id_space)
        return action_space
    
    

