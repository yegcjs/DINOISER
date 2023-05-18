import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass, field
from fairseq.dataclass import FairseqDataclass
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq import metrics

@register_criterion("length_classification_loss")
class LengthClassificationLoss(FairseqCriterion):
    def __init__(self, task):
        super().__init__(task)

    def set_update(self, num_update):
        pass

    def forward(self, model, sample, **unused):
        sample = sample["conditional"]
        encoder_out = model.forward_encoder(sample["net_input"]["src_tokens"])
        target = (sample["target"] != self.task.tgt_dict.pad()).sum(-1)
        preprocessed_target = model.length_predictor.preprocess_target(target, encoder_out)
        logits = model.forward_length(encoder_out, length_beam=1)["model_out"]
        loss = F.cross_entropy(logits, preprocessed_target)
        # print(logits.shape, preprocessed_target.shape)
        # print((~encoder_out["padding_mask"]).sum(-1)[:10], target[:10], preprocessed_target[:10])
        logging_output = {
            "nsentences": target.size(0),
            "ntokens": target.sum().item(),
            "loss": loss.data
        }
        return loss, 0, logging_output
    
    @staticmethod
    def reduce_metrics(logging_outputs):
        nsentences = [log["nsentences"] for log in logging_outputs]
        loss = [log["loss"] for log in logging_outputs]
        loss = sum([n*l for n, l in zip(nsentences, loss)]) / sum(nsentences)
        metrics.log_scalar("loss", loss)