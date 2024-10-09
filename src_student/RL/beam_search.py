import torch
import numpy as np
import torch.nn as nn

EPSILON = float(np.finfo(float).eps)

def safe_log(x):
    return torch.log(x + EPSILON)

def beam_search(policy_network, head, question_emb, kg):
    def batch_lookup(M, idx, vector_output=True):
        batch_size, w = M.size()
        batch_size2, sample_size = idx.size()
        assert(batch_size == batch_size2)

        if sample_size == 1 and vector_output:
            samples = torch.gather(M, 1, idx).view(-1)
        else:
            samples = torch.gather(M, 1, idx)
        return samples
    
    def pad_and_cat(a, padding_value, padding_dim=1):
        max_dim_size = max([x.size()[padding_dim] for x in a])
        padded_a = []
        for x in a:
            if x.size()[padding_dim] < max_dim_size:
                res_len = max_dim_size - x.size()[1]
                pad = nn.ConstantPad1d((0, res_len), padding_value)
                padded_a.append(pad(x))
            else:
                padded_a.append(x)
        return torch.cat(padded_a, dim=0)
    
    def pad_and_cat_action_space(action_spaces, inv_offset):
        db_r_space, db_e_space, db_action_mask , db_tripleID_space = [], [], [], []
        for (r_space, e_space), action_mask, triple_id in action_spaces:
            db_r_space.append(r_space)
            db_e_space.append(e_space)
            db_action_mask.append(action_mask)
            db_tripleID_space.append(triple_id)
        r_space = pad_and_cat(db_r_space, padding_value=0)[inv_offset]
        e_space = pad_and_cat(db_e_space, padding_value=0)[inv_offset]
        action_mask = pad_and_cat(db_action_mask, padding_value=0)[inv_offset]
        tripleID_space = pad_and_cat(db_tripleID_space, padding_value=0)[inv_offset]
        action_space = ((r_space, e_space), action_mask, tripleID_space)
        return action_space

    def top_k_action_and_flat(log_action_dist, action_space):
        full_size = len(log_action_dist)
        (r_space, e_space), _, _ = action_space
        origin_action_space_size = r_space.size()[1]
        log_action_dist = log_action_dist.view(batch_size, -1)
        beam_action_space_size = log_action_dist.size()[1]
        # beam_action_space_size is the bucket size
        k = min(20, beam_action_space_size)
        log_action_prob, action_ind = torch.topk(log_action_dist, k)
        next_e = batch_lookup(e_space.view(batch_size, -1), action_ind)
        log_action_prob = log_action_prob
        return next_e, log_action_prob

    batch_size = len(head)

    db_outcomes, inv_offset, _, _ = policy_network.transit(
            head, question_emb, kg, use_action_space_bucketing=True)
    db_action_dist = []
    db_action_spaces = []
    for _, action_dist in db_outcomes:
        db_action_dist.append(action_dist)
        db_action_spaces.append(_)
    action_space = pad_and_cat_action_space(db_action_spaces, inv_offset)
    action_dist = pad_and_cat(db_action_dist, padding_value=0)[inv_offset]
    db_outcomes = [(action_space, action_dist)]
    inv_offset = None

    action_space, action_dist = db_outcomes[0]
    log_action_dist = safe_log(action_dist)
    nodes, log_action_prob = top_k_action_and_flat(log_action_dist, action_space)
    return nodes, log_action_prob