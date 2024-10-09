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
from abc import abstractmethod


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
        x = x.squeeze(1)

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

        x = x.unsqueeze(1)

        return x



class BaseVAE(nn.Module):
    
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input):
        raise NotImplementedError

    def decode(self, input):
        raise NotImplementedError

    def sample(self, batch_size:int, current_device: int, **kwargs):
        raise NotImplementedError

    def generate(self, x, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs):
        pass

    @abstractmethod
    def loss_function(self, *inputs, **kwargs):
        pass


class QaVAE(BaseVAE):

    def __init__(self, question_dim, latent_dim, relation_num, pretrained_relation_embeddings, device):
        super(QaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.device = device

        self.roberta_pretrained_weights = 'roberta-base'
        self.roberta_model = RobertaModel.from_pretrained(self.roberta_pretrained_weights)
        #self.roberta_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

        self.fc_mu = nn.Linear(question_dim, latent_dim)
        self.fc_var = nn.Linear(question_dim, latent_dim)
        nn.init.xavier_uniform(self.fc_mu.weight)
        nn.init.xavier_uniform(self.fc_var.weight)

        self.decoder_SOS = nn.Embedding(2, latent_dim + relation_num)
        nn.init.xavier_uniform(self.decoder_SOS.weight)

        self.n_layers = 1
        self.GRU_decoder = nn.LSTM(self.latent_dim + relation_num, self.latent_dim + relation_num, self.n_layers, bidirectional=False, batch_first=True)

        for name, param in self.GRU_decoder.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

        self.relation_num = relation_num
        self.fc_out = MulticlassClassification(self.latent_dim + relation_num, relation_num)

        for name, param in self.fc_out.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)


        self.pretrained_relation_embeddings = pretrained_relation_embeddings

        self.loss = torch.nn.BCELoss(reduction='sum')

        # don't forget to initialize model
        


    def getQuestionEmbedding(self, question_tokenized, attention_mask):
        roberta_last_hidden_states = self.roberta_model(question_tokenized, attention_mask=attention_mask)[0]
        states = roberta_last_hidden_states.transpose(1,0)
        cls_embedding = states[0]
        question_embedding = cls_embedding
        # question_embedding = torch.mean(roberta_last_hidden_states, dim=1)
        return question_embedding

    def encode(self, question_tokenized, attention_mask):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        question_embedding = self.getQuestionEmbedding(question_tokenized, attention_mask)
        # question_embedding = torch.mean(roberta_last_hidden_states, dim=1)
        
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(question_embedding)
        log_var = self.fc_var(question_embedding)

        return [mu, log_var]
    

    def attention_path_decoder2(self, z, initial_hidden=None, teacher_forcing_ratio=0.5, groundtruth_path=None, sample_num=1):
        
        path_class_prob_prediction_list = []
        
        if initial_hidden is None:
            decoder_hidden_idx = torch.zeros(1, z.size()[0], dtype=torch.long).to(self.device)
            decoder_cell_idx = torch.ones(1, z.size()[0], dtype=torch.long).to(self.device)

            decoder_hidden = self.decoder_SOS(decoder_hidden_idx)
            decoder_celll = self.decoder_SOS(decoder_cell_idx)
        else:
            decoder_hidden, decoder_celll = initial_hidden

        STEP = 1
        decoder_input = z.unsqueeze(1)
        predicted_relations_emb = []
        path = []
        for t in range(STEP):
            output, (hidden, cell) = self.GRU_decoder(decoder_input, (decoder_hidden, decoder_celll))
            # predict according to output
            prediction = self.fc_out(output).squeeze(1)
            path_class_prob_prediction_list.append(prediction)
            # prediction to next input
            if groundtruth_path is not None:
                teacher_force = random.random() < teacher_forcing_ratio
                top1 = prediction.argmax(1)
                decoder_input_idx = groundtruth_path[:,t] if teacher_force else top1
            else:
                decoder_input_idx = prediction.argmax(1)
            decoder_input = self.pretrained_relation_embeddings(decoder_input_idx)
            
            predicted_relations_emb.append(decoder_input)
            decoder_hidden = hidden
            decoder_celll = cell
        return predicted_relations_emb, path_class_prob_prediction_list
    

    def attention_path_decoder(self, z, initial_hidden=None, teacher_forcing_ratio=0.5, groundtruth_path=None, sample_num=1):
        
        path_class_prob_prediction_list = []
        
        if initial_hidden is None:
            decoder_hidden_idx = torch.zeros(1, z.size()[0], dtype=torch.long).to(self.device)
            decoder_cell_idx = torch.ones(1, z.size()[0], dtype=torch.long).to(self.device)

            decoder_hidden = self.decoder_SOS(decoder_hidden_idx)
            decoder_celll = self.decoder_SOS(decoder_cell_idx)
        else:
            decoder_hidden, decoder_celll = initial_hidden

        STEP = 1
        decoder_input = z.unsqueeze(1)
        predicted_relations_emb = []
        path = []
        for t in range(STEP):
            output, (hidden, cell) = self.GRU_decoder(decoder_input, (decoder_hidden, decoder_celll))
            # predict according to output
            prediction = self.fc_out(output).squeeze(1)
            path_class_prob_prediction_list.append(prediction)
            # prediction to next input

            _, decoder_input_idx = torch.topk(prediction, sample_num)

            decoder_input = self.pretrained_relation_embeddings(decoder_input_idx)
            
            predicted_relations_emb.append(decoder_input)
            decoder_hidden = hidden
            decoder_celll = cell
        return predicted_relations_emb, path_class_prob_prediction_list

    def decode(self, z, neighbor_rel_onehot, sample_num=1):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        z = torch.cat((z, neighbor_rel_onehot), 1)
        pathes_embeddings, path_prediction_list = self.attention_path_decoder(z, sample_num=sample_num)
        return pathes_embeddings, path_prediction_list

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        #std = torch.exp(0.5 * logvar)
        #eps = torch.randn_like(std)
        #return eps * std + mu
        return mu

    def forward(self, question_tokenized, attention_mask, neighbor_rel_onehot, sample_num, **kwargs):
        mu, log_var = self.encode(question_tokenized, attention_mask)
        z = self.reparameterize(mu, log_var)
        pathes_embeddings, path_prediction_list = self.decode(z, neighbor_rel_onehot, sample_num)
        return  pathes_embeddings, path_prediction_list, mu, log_var

    def loss_function(self, pathes_embeddings, path_prediction_list, p_rels_groundtruth, mu, log_var):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """        
        total_loss = 0
        for i in range(p_rels_groundtruth.shape[1]):
            path_pred = path_prediction_list[i]
            # path_loss
            tmp_path = p_rels_groundtruth[:,i].view(-1)
            path_ground_truth = F.one_hot(tmp_path, num_classes=self.relation_num)
            path_loss = self.loss(path_pred, path_ground_truth.float())
            total_loss = total_loss + path_loss


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        kld_weight = 1 # Account for the minibatch samples from the dataset

        loss = total_loss #+ kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':total_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self,
               num_samples,
               current_device, **kwargs):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x, **kwargs):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]



