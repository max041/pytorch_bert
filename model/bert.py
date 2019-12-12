import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.transformer_encoder import Encoder


class BertConfig(object):
    def __init__(self, config_path):
        self._config_dict = self._parse(config_path)

    def _parse(self, config_path):
        try:
            with open(config_path) as json_file:
                config_dict = json.load(json_file)
        except Exception:
            raise IOError("Error in parsing bert model config file '%s'" %
                          config_path)
        else:
            return config_dict

    def __getitem__(self, key):
        return self._config_dict[key]

    def print_config(self):
        for arg, value in sorted(self._config_dict.items()):
            print('%s: %s' % (arg, value))
        print('------------------------------------------------')


class BertModel(nn.Module):
    def __init__(self,
                 config,
                 weight_sharing=True,
                 use_fp16=False,
                 enable_deterministic_mode=False):
        super(BertModel, self).__init__()
        self._emb_size = config['hidden_size']
        self._n_layer = config['num_hidden_layers']
        self._n_head = config['num_attention_heads']
        self._voc_size = config['vocab_size']
        self._max_position_seq_len = config['max_position_embeddings']
        self._sent_types = config['type_vocab_size']
        self._hidden_act = config['hidden_act']
        self._prepostprocess_dropout = config['hidden_dropout_prob']
        self._attention_dropout = config['attention_probs_dropout_prob']

        self._weight_sharing = weight_sharing

        # TODO
        # Initialize all weights by truncated normal initializer, and all biases
        # will be initialized by constant zero by default.
        # self._param_initializer = TruncatedNormal(config['initializer_range'])

        self.word_embedding = nn.Embedding(self._voc_size, self._emb_size)
        self.pos_embedding = nn.Embedding(self._max_position_seq_len, self._emb_size)
        self.sent_embedding = nn.Embedding(self._sent_types, self._emb_size)

        self.pre_encoder_layer_norm = nn.LayerNorm(self._emb_size)

        self.encoder = Encoder(n_layer=self._n_layer,
                               n_head=self._n_head,
                               d_key=self._emb_size // self._n_head,
                               d_value=self._emb_size // self._n_head,
                               d_model=self._emb_size,
                               d_inner_hid=self._emb_size * 4,
                               prepostprocess_dropout=self._prepostprocess_dropout,
                               attention_dropout=self._attention_dropout,
                               relu_dropout=0,
                               hidden_act=self._hidden_act)

    def forward(self,
                src_ids,
                position_ids,
                sentence_ids,
                input_mask):
        emb_out = self.word_embedding(src_ids)
        position_emb_out = self.pos_embedding(position_ids)
        sent_emb_out = self.sent_embedding(sentence_ids)

        emb_out = emb_out + position_emb_out
        emb_out = emb_out + sent_emb_out
        emb_out = self.pre_encoder_layer_norm(emb_out)
        if self._prepostprocess_dropout:
            emb_out = F.dropout(emb_out, self._prepostprocess_dropout)

        with torch.no_grad():
            self_attn_mask = torch.matmul(input_mask, input_mask.transpose(1, 2))
            n_head_self_attn_mask = 1000 * (self_attn_mask - 1)

        return self.encoder(emb_out, n_head_self_attn_mask)



if __name__ == '__main__':
    import os
    # print(os.path.join(os.getcwd(), '..', 'configs', 'bert_large.json'))
    bc = BertConfig(os.path.join(os.getcwd(), '..', 'configs', 'bert_large.json'))
    bc.print_config()
    # print(bc)