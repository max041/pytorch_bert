import torch.nn as nn
import torch.nn.functional as F





class MultiHeadAttention(nn.Module):
    def __init__(self,
                 d_key,
                 d_value,
                 d_model,
                 n_head=1,
                 droput_rate=0.):
        """
        Multi-Head Attention. Note that attn_bias is added to the logit before
        computing softmax activiation to mask certain selected positions so that
        they will not considered in attention weights.
        :param d_key:
        :param d_value:
        :param d_model:
        :param n_head:
        :param droput_rate:
        """
        super(MultiHeadAttention, self).__init__()
        self._d_key = d_key
        self._d_value = d_value
        self._d_model = d_model
        self._n_head = n_head
        self._dropout_rate = droput_rate

        self.query_fc = nn.Linear()

    def forward(self, queries, keys, values, attn_bias):
        pass


class EncoderLayer(nn.Module):
    def __init__(self,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 prepostprocess_dropout,
                 attention_dropout,
                 relu_dropout,
                 hidden_act):
        ''' # TODO give more comprehensive explanation.
        The encoder layer that can be stacked to form a deep encoder.
        This module consists of a multi-head (self) attention followed by
        position-wise feed-forward networks and both the two components to gather
        with the post_process_layer to add residual connection, layer normalization
        and dropout.

        :param n_head:
        :param d_key:
        :param d_value:
        :param d_model:
        :param d_inner_hid:
        :param prepostprocess_dropout:
        :param attention_dropout:
        :param relu_dropout:
        :param hidden_act:
        '''
        super(EncoderLayer, self).__init__()
        self._n_head = n_head
        self._d_key = d_key
        self._d_value = d_value
        self._d_model = d_model
        self._d_inner_hid = d_inner_hid
        self._prepostprocess_dropout = prepostprocess_dropout
        self._attention_dropout = attention_dropout
        self._relu_dropout = relu_dropout
        self._hidden_act = hidden_act

        self.multi_head_attention = MultiHeadAttention(d_key, d_value, d_model, n_head, attention_dropout)
        self.encoder_layer_norm = nn.LayerNorm(self._d_model)
        self.ffn_fc_0 = nn.Linear(self._d_model, self._d_inner_hid)
        self.ffn_fc_1 = nn.Linear(self._d_inner_hid, self._d_model)
        self.encoder_post_ffn_layer_norm = nn.LayerNorm(self._d_model)

    def forward(self, enc_input, attn_bias):
        attn_output = self.multi_head_attention(enc_input, None, None, attn_bias)
        if self._prepostprocess_dropout:
            out = F.dropout(attn_output, self._prepostprocess_dropout)
        else:
            out = attn_output
        out = out + enc_input
        attn_output = self.encoder_layer_norm(out)

        hidden = self.ffn_fc_0(attn_output)
        hidden = getattr(F, self._hidden_act)(hidden)
        if self._relu_dropout:
            hidden = F.dropout(hidden, self._relu_dropout)
        ffd_output = self.ffn_fc_1(hidden)

        if self._prepostprocess_dropout:
            out = F.dropout(ffd_output, self._prepostprocess_dropout)
        else:
            out = ffd_output
        out = out + attn_output
        return self.encoder_post_ffn_layer_norm(out)


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
        '''
        The encoder is composed of a stack of identical layers returned by calling
        EncoderLayer.
        :param n_layer:
        :param n_head:
        :param d_key:
        :param d_value:
        :param d_model:
        :param d_inner_hid:
        :param prepostprocess_dropout:
        :param attention_dropout:
        :param relu_dropout:
        :param hidden_act:
        '''
        super(Encoder, self).__init__()
        self._n_layer = n_layer

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(n_head, d_key, d_value, d_model, d_inner_hid,
                         prepostprocess_dropout, attention_dropout, relu_dropout, hidden_act)
            for _ in range(self._n_layer)
        ])

    def forward(self, enc_input, attn_bias):
        for i in range(self._n_layer):
            enc_output = self.encoder_layers[i](enc_input, attn_bias)
            enc_input = enc_output
        return enc_input


