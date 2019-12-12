import torch.nn as nn
import torch.nn.functional as F





class MultiHeadAttention(nn.Module):
    def __init__(self,
                 d_key,
                 d_value,
                 d_model,
                 n_head=1,
                 droput_rate=0.,
                 cache=None,
                 param_initializer=None,
                 name='multi_head_att'):
        """
        Multi-Head Attention. Note that attn_bias is added to the logit before
        computing softmax activiation to mask certain selected positions so that
        they will not considered in attention weights.
        :param d_key:
        :param d_value:
        :param d_model:
        :param n_head:
        :param droput_rate:
        :param cache:
        :param param_initializer:
        :param name:
        """
        super(MultiHeadAttention, self).__init__()
        self.d_key = d_key
        self.d_value = d_value
        self.d_model = d_model


        self.query_fc = nn.Linear()

    def forward(self, queries, keys, values, attn_bias):
        pass


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()

    def forward(self):
        pass


class Encoder(nn.Module):
    def __init__(self,
                 n_layer,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 prepostprocess_dropout,
                 attention_dropout,
                 relu_dropout,
                 hidden_act):
        super(Encoder, self).__init__()

    def forward(self):
        pass