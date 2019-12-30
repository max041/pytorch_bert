import torch
import torch.nn as nn
import torch.nn.functional as F
from model.bert import BertModel


class Classifier(nn.Module):
    def __init__(self,
                 bert_config,
                 num_labels):
        super(Classifier, self).__init__()
        self.bert = BertModel(bert_config)
        self.cls_out = nn.Linear(bert_config['hidden_size'], num_labels)

    def forward(self,
                src_ids,
                position_ids,
                sentence_ids,
                input_mask,
                labels):
        cls_feats = self.bert(src_ids, position_ids, sentence_ids, input_mask, pooled=True)
        cls_feats = F.dropout(cls_feats, 0.1)
        logits = self.cls_out(cls_feats)

        with torch.no_grad():
            probs = F.softmax(logits, 1)
            accuracy = (labels == probs.topk(1, 1)[1]).type(torch.float32).sum() / len(labels)

        ce_loss = F.cross_entropy(logits, labels.reshape(-1), reduction='none')
        loss = ce_loss.mean()
        return loss, probs, accuracy


