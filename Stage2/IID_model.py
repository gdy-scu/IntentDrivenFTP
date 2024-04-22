# coding=utf-8
import torch
import torch.nn as nn
from transformers import BertModel

class BERT4IntentDetection(nn.Module):
    def __init__(self, config):
        super(BERT4IntentDetection, self).__init__()
        self.config = config

        self.backbone = BertModel.from_pretrained(config.Pretrained_model_path)
        for param in self.backbone.parameters():
            param.requires_grad = True

        self.classifier = nn.Sequential(nn.Dropout(0.2),
                                        nn.Linear(config.bert_out_dim, config.num_class),
                                        nn.Sigmoid()
                                        )

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):

        outputs = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        emb = outputs.last_hidden_state.sum(dim=1)
        logits = self.classifier(emb)

        return emb, logits