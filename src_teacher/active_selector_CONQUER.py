import logging
from sklearn.cluster import KMeans
import os
import numpy as np
import random
import json
from collections import defaultdict

log = logging.getLogger()



# this one is the base class; parent class
class ActiveDataSelector():
    def __init__(self, conversation_path, config):
        self.conversation_path = conversation_path
        self.sampled_data = []
        self.remaining_data = []
        self.have_not_initialized = True
        self.config = config
        self._init()
    
    def next(self):
        pass
    
    def _init(self):
        # read conversation
        with open(self.conversation_path, "r") as data:
            self.remaining_data = json.load(data)
    
    def process(self, data):
        data_list = []
        for conv in data:
            tmp = []
            anchor = conv['seed_entity_text']
            tmp.append(anchor)
            questions =  conv['questions']
            previous_answer = anchor
            for q in questions:
                question_txt = q['question']
                reformulation = q['reformulations']
                ########### reformulation
                if not self.config.is_cal_p1:
                    THRESHOLD = 1
                    rfs = []
                    rfs.append(question_txt)
                    if len(reformulation) == 0:
                        for idx in range(THRESHOLD):
                            rfs.append(question_txt)
                    elif len(reformulation) >= THRESHOLD:
                        reformulation = reformulation[0:THRESHOLD]
                        for e in reformulation:
                            rfs.append(e['reformulation'])
                    else:
                        reformulation = reformulation * THRESHOLD
                        reformulation = reformulation[0:THRESHOLD]
                        for e in reformulation:
                            rfs.append(e['reformulation'])
                    tmp.append(rfs)
                else:
                    # this part of code is for calculate p@1
                    rfs = []
                    if self.config.cal_p1_step == 0:
                        rfs.append(question_txt)
                    elif self.config.cal_p1_step == 1:
                        if len(reformulation) == 0:
                            rfs.append(question_txt)
                        else:
                            rfs.append(reformulation[0]['reformulation'])
                    elif self.config.cal_p1_step == 2:
                        if len(reformulation) == 0:
                            rfs.append(question_txt)
                        elif len(reformulation) <= 2:
                            rfs.append(reformulation[-1]['reformulation'])
                        else:
                            rfs.append(reformulation[1]['reformulation'])
                    elif self.config.cal_p1_step == 3:
                        if len(reformulation) == 0:
                            rfs.append(question_txt)
                        elif len(reformulation) <= 3:
                            rfs.append(reformulation[-1]['reformulation'])
                        else:
                            rfs.append(reformulation[2]['reformulation'])
                    elif self.config.cal_p1_step == 4:
                        if len(reformulation) == 0:
                            rfs.append(question_txt)
                        elif len(reformulation) <= 4:
                            rfs.append(reformulation[-1]['reformulation'])
                        else:
                            rfs.append(reformulation[3]['reformulation'])
                    
                    tmp.append(rfs)


                answer = q['gold_answer_text']
                #tmp.append(rfs)
                tmp.append(answer)
                # for each question, add its corresponding triple
                # if can be reached from head or not
                # check whether answer is adj to previous_answer
            tmp.append(0)
            previous_answer = questions[0]['gold_answer_text']
            for q in questions[1:]:
                answer = q['gold_answer_text']
                if answer in self.config.adj_map_no_rel[anchor]:
                    tmp.append(0)
                else:
                    if answer in self.config.adj_map_no_rel[previous_answer]:
                        tmp.append(1)
                    else:
                        tmp.append(0)
                    #if answer not in self.config.adj_map_no_rel[previous_answer]:
                    #    print(anchor, question_txt, answer)
                previous_answer = answer
            # for q in questions:
            #     # should also append the reformulation of each question
            #     question_txt = q['question']
            #     reformulation = q['reformulations']
            #     THRESHOLD = 2
            #     rfs = []
            #     if len(reformulation) == 0:
            #         for idx in range(THRESHOLD):
            #             rfs.append(question_txt)
            #     elif len(reformulation) >= THRESHOLD:
            #         reformulation = reformulation[0:THRESHOLD]
            #         for e in reformulation:
            #             rfs.append(e['reformulation'])
            #     else:
            #         reformulation = reformulation * THRESHOLD
            #         reformulation = reformulation[0:THRESHOLD]
            #         for e in reformulation:
            #             rfs.append(e['reformulation'])
            #     tmp.append(rfs)
            data_list.append(tmp)
        return data_list



class ActiveRandomSelector(ActiveDataSelector):

    def __init__(self, conversation_path, sample_size, config):
        super().__init__(conversation_path, config)
        self.sample_size = sample_size
        self.config = config
    
    # different sample strategy
    def _next(self):
        current_sample, self.remaining_data = self.remaining_data[:self.sample_size], self.remaining_data[self.sample_size:]
        self.sampled_data.extend(current_sample)
        return self.sampled_data
    
    def next(self):
        data = self._next()
        # change data
        return self.process(data)

class ActiveKMeansSelector(ActiveDataSelector):

    def __init__(self, conversation_path, entity_emb, entity_dict, sample_size, cluster_num, config):
        self.sample_size = sample_size
        self.cluster_num = cluster_num
        self.entity_dict = entity_dict
        self.kg_node_2_id_map = entity_dict
        self.kg_node_embeddings = entity_emb
        self.clusters = defaultdict(list)
        self.config = config
        super().__init__(conversation_path, config)

    def _init(self):
        super(ActiveKMeansSelector, self)._init() 
        self.convID_2_conv = {}
        for conv in self.remaining_data:
            c_id = conv['conv_id']
            assert (c_id not in self.convID_2_conv)
            self.convID_2_conv[c_id] = conv

        self.build_clusters()
        
    
    def _make_conversation_embedding(self):
        def get_kg_entities(conv):
            anchor = conv['seed_entity_text']
            others = []
            questions =  conv['questions']
            for q in questions:
                answer = q['gold_answer_text']
                others.append(answer)
            others.append(anchor)
            return others

        conv_emb_map = {}
        conv_emb_list = []
        idx_to_convID = {}
        for conv in self.remaining_data:
            # get its anchor node and answer nodes' embedding
            all_nodes = get_kg_entities(conv)
            # get average embedding
            ebs = []
            for n in all_nodes:
                kg_node_id = self.kg_node_2_id_map[n]
                ebs.append(self.kg_node_embeddings[kg_node_id])
            # calculate average
            conv_emb = np.average(ebs, axis=0)
            assert (conv['conv_id'] not in conv_emb_map)
            conv_emb_map[conv['conv_id']] = conv_emb
            conv_emb_list.append(conv_emb)
            idx_to_convID[len(idx_to_convID)] = conv['conv_id']
        
        self.emb_idx_2_convID = idx_to_convID
        self.conv_emb = np.array(conv_emb_list)
        return idx_to_convID, conv_emb_list


    def do_clusterize(self, q_embs, cluster_num):
        
        labels = {}
        # calculate the embedding of conversation
        # 
        kmeans = KMeans(n_clusters=cluster_num).fit(q_embs)
        labels_lst = kmeans.labels_.tolist()
        for entity_id, cluster_id in enumerate(labels_lst):
                labels[entity_id] = cluster_id
        return labels

    def build_clusters(self):
        log.info("Clustering: started")

        idx_to_convID, q_embs = self._make_conversation_embedding() 
        # labels[entity_id] = cluster_id
        qID2cluster = self.do_clusterize(q_embs, self.cluster_num)
        # iterate through conversation and 
        for qid, cluster_id in qID2cluster.items():
            convID = idx_to_convID[qid]
            self.clusters[cluster_id].append(self.convID_2_conv[convID])

        log.info("Clustering: finished")

    def _next(self):
        if self.have_not_initialized:
            self.have_not_initialized = False
            a = self.init_w_clustering()
            self.sampled_data.extend(a)
            return self.sampled_data
        else:
            a = self.update_clustering()
            self.sampled_data.extend(a)
            return self.sampled_data
    
    def next(self):
        data = self._next()
        return self.process(data)
            
    def init_w_clustering(self):
        empty_clusters = []
        initial_sample = []
        conversation_per_cluster = int(
            round(
                self.sample_size / len(self.clusters)
            )
        )

        if conversation_per_cluster == 0:
            conversation_per_cluster = 1
        
        stop_sampling = False

        for cluster_id, cluster_data in self.clusters.items():
            if stop_sampling:
                end_index = 0
            else:
                end_index = min(conversation_per_cluster, len(cluster_data))
            random.shuffle(cluster_data)
            initial_sample.extend(cluster_data[:end_index])

            if len(cluster_data) - end_index > 1:
                self.clusters[cluster_id] = cluster_data[end_index:]
            else:
                empty_clusters.append(cluster_id)
            
            if len(initial_sample) == self.sample_size:
                stop_sampling = True
        
        for cluster_id in empty_clusters:
            self.clusters.pop(cluster_id)
        
        return initial_sample
    
    
    # for k-means, at the begining, sample same amount of data points in each cluster
    # then, sample according to ratio
    def update_clustering(self):
        empty_clusters = []
        current_sample = []
        all_clusters_size = sum(len(v) for v in self.clusters.values())

        for cluster_id, cluster_data in self.clusters.items():
            random.shuffle(cluster_data)
            current_cluster_ratio = float(len(cluster_data)) / all_clusters_size
            n = int(round(current_cluster_ratio * self.sample_size))

            if n == 0:
                n = 1

            current_sample.extend(cluster_data[:n])

            if len(cluster_data) - n > 1:
                self.clusters[cluster_id] = cluster_data[n:]
            else:
                empty_clusters.append(cluster_id)
            
        for cluster_id in empty_clusters:
            self.clusters.pop(cluster_id)
        
        return current_sample



