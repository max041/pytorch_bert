import torch
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

        self.query_fc = nn.Linear(self._d_model, self._d_key * self._n_head)
        self.key_fc = nn.Linear(self._d_model, self._d_key * self._n_head)
        self.value_fc = nn.Linear(self._d_model, self._d_value * self._n_head)
        self.output_fc = nn.Linear(self._d_value * self._n_head, self._d_model)

    def __compute_qkv(self, queries, keys, values):
        '''
        Add linear projection to queries, keys and values.
        :param queries:
        :param keys:
        :param values:
        :param n_head:
        :param d_key:
        :param d_value:
        :return:
        '''
        q = self.query_fc(queries)
        k = self.key_fc(keys)
        v = self.value_fc(values)
        return q, k, v

    def __split_heads(self, x):
        '''
        Reshape the last dimension of input tensor x so that it becomes two
        dimensions and then transpose. Specifically, input a tensor with shape
        [bs, max_sequence_length, n_head * hidden_dim].
        :param x:
        :return:
        '''
        hidden_size = x.shape[-1]
        x_shape = x.shape
        reshaped = x.reshape([x_shape[0], x_shape[1], self._n_head, hidden_size // self._n_head])

        # permute the dimensions into:
        # [batch_size, n_head, max_sequence_len, hidden_size_per_head]
        return reshaped.transpose(1, 2)

    def __combine_heads(self, x):
        '''
        Transpose and then reshape the last two dimensions of input tensor x
        so that it becomes one dimension, which is reverse to __split_heads.
        :param x:
        :return:
        '''
        if len(x.shape) == 3: return x
        if len(x.shape) != 4:
            raise ValueError('Input(x) should be a 4-D Tensor.')

        trans_x = x.transpose(1, 2)
        x_shape = trans_x.shape
        return trans_x.reshape([x_shape[0], x_shape[1], -1])

    def scaled_dot_product_attention(self, q, k, v, attn_bias):
        '''
        Scaled Dot-Product Attention
        :param k:
        :param v:
        :param attn_bias:
        :return:
        '''
        scaled_q = (self._d_key**-0.5) * q
        product = torch.matmul(scaled_q, k.transpose(2, 3))
        if attn_bias:
            product += attn_bias
        weights = F.softmax(product, dim=-1)
        if self._dropout_rate:
            weights = F.dropout(weights, self._dropout_rate)
        out = torch.matmul(weights, v)
        return out

    def forward(self, queries, keys, values, attn_bias):
        keys = queries if keys is None else keys
        values = keys if values is None else values

        if not (len(queries.shape) == len(keys.shape) == len(values.shape) == 3):
            raise ValueError(
                'Inputs: queries, keys and values should all be 3-D tensors'
            )

        q, k, v = self.__compute_qkv(queries, keys, values)

        q = self.__split_heads(q)
        k = self.__split_heads(k)
        v = self.__split_heads(v)

        ctx_multiheads = self.scaled_dot_product_attention(q, k, v, attn_bias)

        out = self.__combine_heads(ctx_multiheads)

        # Project back to the model size.
        proj_out = self.output_fc(out)
        return proj_out



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


