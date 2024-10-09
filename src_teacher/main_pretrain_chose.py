# coding=utf-8
from dataloader import *
from model_chose import *
import argparse
from torch.optim.lr_scheduler import ExponentialLR
from config_train_chose import Config
from active_selector_CONQUER import *
import sys

sys.path.append("./")

from model_LSTM import *

from logger import setup_logger

"""
This one is used to pretrain the topic entity selection model (model_LSTM.py)
But because the model will choose the global topic entity most of the time, 
in RL based model, we simplely use the global topic entity for each input query
"""


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--al-epochs", dest="al_epochs", type=int, help="iterations of active learning")
    parser.add_argument("--batch-size", dest="batch_size", type=int)
    parser.add_argument("--dataset", dest="dataset")
    parser.add_argument("--embedding-dim", dest="embedding_dim", type=int)
    parser.add_argument("--early-stop-threshold", dest="early_stop_threshold", type=int)
    parser.add_argument("--eval-rate", dest="eval_rate", type=int, help="make evaluation each n epochs")
    parser.add_argument("--inner-lr", dest="inner_learning_rate", type=int)
    parser.add_argument("--lr", dest="learning_rate", type=float)
    parser.add_argument("--lr-decay", dest="learning_rate_decay", type=float)
    parser.add_argument("--model", dest="model_name", choices=["ConvE", "MLP"])
    parser.add_argument("--n-clusters", dest="n_clusters", type=int)
    parser.add_argument("--sample-size", dest="sample_size", type=int, help="number of training examples per one AL iteration")
    parser.add_argument("--sampling-mode", dest="sampling_mode", choices=["random", "uncertainty", "structured", "structured-uncertainty"])
    parser.add_argument("--training-mode", dest="training_mode", choices=["retrain", "incremental", "meta-incremental"])
    parser.add_argument("--window-size", dest="window_size", type=int)

    return parser.parse_args()




def validate_v2(config, model, data_loader, device, num_entities, output_rank=False):
    model.eval()
    config.training = False

    hits = [[] for _ in range(10)]
    ranks = []

    top_results = []

    loader = tqdm(data_loader, total=len(data_loader), unit="batches")
    total_error = 0
    for i_batch, data in enumerate(loader):
        head = data[0].to(device)
        q1 = data[1].to(device)
        m1 = data[2].to(device)
        a1 = data[3]
        q2 = data[4].to(device)
        m2 = data[5].to(device)
        a2 = data[6]
        q3 = data[7].to(device)
        m3 = data[8].to(device)
        a3 = data[9]
        q4 = data[10].to(device)
        m4 = data[11].to(device)
        a4 = data[12]
        q5 = data[13].to(device)
        m5 = data[14].to(device)
        a5 = data[15]
        q_text1 = data[16]
        q_text2 = data[17]
        q_text3 = data[18]
        q_text4 = data[19]
        q_text5 = data[20]
        answers = model.get_score_ranked(head, q1, m1,
        q2, m2,
        q3, m3,
        q4, m4,
        q5, m5,
        data[21].to(device), data[22].to(device), data[23].to(device), data[24].to(device), data[25].to(device)
        )
        total_error = total_error + answers

    print(total_error)
    return total_error

def output_top(config, model, data_loader, device, num_entities):
    validate_v2(config=config, model=model, data_loader = data_loader, device=device, num_entities=num_entities, output_rank=True)

def train_again(config, model, optimizer, scheduler, dataloader, valid_data_loader, test_data_loader, device, num_entities, max_acc):
    validate_v2(config=config, model=model, data_loader = test_data_loader, device=device, num_entities=num_entities, output_rank=True)

    best_hit1 = max_acc
    best_res = None
    config.training = True
    for epoch in range(config.al_epochs):
        log.info("{} iteration of active learning: started".format(epoch + 1))
        log.info("Train model: started")

        model.train()
        loader = tqdm(dataloader, total=len(dataloader), unit="batches")
        running_loss = 0
        for i_batch, data in enumerate(loader):
            model.zero_grad()
            loss = model(data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device),
            data[4].to(device), data[5].to(device), data[6].to(device),
            data[7].to(device), data[8].to(device), data[9].to(device),
            data[10].to(device), data[11].to(device), data[12].to(device),
            data[13].to(device), data[14].to(device), data[15].to(device),
            data[21].to(device), data[22].to(device), data[23].to(device), data[24].to(device), data[25].to(device)
            )
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loader.set_postfix(Loss=running_loss/((i_batch+1)*config.batch_size), Epoch=epoch)
            loader.set_description('{}/{}'.format(epoch, config.al_epochs))
            loader.update()
        scheduler.step()

        if epoch % 3 == 0 and epoch != 0:
            # evaluate
            model.eval()
            print("evaluate")
            valid_res = validate_v2(config=config, model=model, data_loader = valid_data_loader, device=device, num_entities=num_entities)
            if best_hit1 > valid_res:
                best_hit1 = valid_res
                best_res = valid_res
                # run test
                print("-----------------TEST--------------------------")
                test_res = validate_v2(config=config, model=model, data_loader = test_data_loader, device=device, num_entities=num_entities)
                print(test_res)
                # save model
                if config.save_model:
                    torch.save(model.state_dict(), config.model_save_path)


    
    # print final results
    print("*******************************************************************")
    print(best_res)
    return best_hit1

          
def main():
    args = parse_args()

    config = Config(args)

    # load data
    log.info("Initializing training sample streamer")

    # how many rounds
    
    kg_node_embeddings = np.load(config.entity_emb_path)
    relation_embeddings = np.load(config.relation_emb_path)

    if config.sample_strategy == 'kmeans':
        train_batcher = ActiveKMeansSelector(config.conversation_path, kg_node_embeddings, config.entity2id, 
        config.sample_size, config.cluster_num, config)
    else:
        train_batcher = ActiveRandomSelector(config.conversation_path, config.sample_size, config)
    
    # use a extremaly large value to get all
    valid_batcher = ActiveRandomSelector(config.conversation_valid_path, 999999, config)
    valid_data = valid_batcher.next()
    valid_conv_dataset = DatasetConversation(valid_data, config.entity2id, config.relation2id, config.entity2neighbor_relation, config.ht2relation)
    valid_data_loader = DataLoader(valid_conv_dataset, batch_size=config.test_batch_size, shuffle=False, num_workers=config.num_workers)

    test_batcher = ActiveRandomSelector(config.conversation_test_path, 999999, config)
    test_data = test_batcher.next()
    test_conv_dataset = DatasetConversation(test_data, config.entity2id, config.relation2id, config.entity2neighbor_relation, config.ht2relation)
    test_data_loader = DataLoader(test_conv_dataset, batch_size=config.test_batch_size, shuffle=False, num_workers=config.num_workers)
    
    log.info("Initializing test_rank streamer")

    # initialize model
    kg_node_embeddings = torch.tensor(kg_node_embeddings)
    relation_embeddings = torch.tensor(relation_embeddings)

    if config.parallel:
        device = torch.device("cuda:0")
    else:
        device = torch.device(config.gpu if config.use_cuda else "cpu")

    config.device = device

    ########### read lstm model
    if config.load_lstm_model:
        rl_lstm_model = RL_LSTM(embedding_dim=config.embedding_dim, num_entities = len(config.entity2id), pretrained_embeddings=kg_node_embeddings, 
        freeze=config.freeze, device=config.device, entdrop = config.entdrop, reldrop = config.reldrop, scoredrop = config.scoredrop, 
        l3_reg = config.l3_reg, ls = config.ls, do_batch_norm=config.do_batch_norm)
        rl_lstm_model.load_state_dict(torch.load("./src_active_learning/best_model.pt"))
        config.other_model = rl_lstm_model
        if config.use_cuda:
            config.other_model.to(device)
    ###############################################

    model = RelationExtractor(config, embedding_dim=config.embedding_dim, num_entities = len(config.entity2id), relation_emb = relation_embeddings, 
        pretrained_embeddings=kg_node_embeddings, 
        freeze=config.freeze, device=config.device, entdrop = config.entdrop, reldrop = config.reldrop, scoredrop = config.scoredrop, 
        l3_reg = config.l3_reg, ls = config.ls, do_batch_norm=config.do_batch_norm)
    
    if config.load_model:
        model.load_state_dict(torch.load(config.model_save_path))

    if torch.cuda.device_count() > 1 and config.parallel:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    if config.use_cuda:
        model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = ExponentialLR(optimizer, config.decay)
    optimizer.zero_grad()

    # each round use action learning to get some new data

    if config.reranking:
        train_batcher = ActiveRandomSelector(config.conversation_test_path, 999999)
        new_data = train_batcher.next()
        if len(new_data) <= 0:
            print("end")
            return
        conv_dataset = DatasetConversation(new_data, config.entity2id, config.relation2id, config.entity2neighbor_relation, config.ht2relation)
        data_loader = DataLoader(conv_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
        output_top(config, model, data_loader, device, len(config.entity2id))
    else:
        max_acc = 99999
        for i in range(config.active_round):
            print("###################################################")
            new_data = train_batcher.next()
            if len(new_data) <= 0:
                print("end")
                break
            conv_dataset = DatasetConversation(new_data, config.entity2id, config.relation2id, config.entity2neighbor_relation, config.ht2relation)
            data_loader = DataLoader(conv_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
            max_acc = train_again(config, model, optimizer, scheduler, data_loader, valid_data_loader, test_data_loader, device, len(config.entity2id), max_acc)
    
    # test data
        

main()



