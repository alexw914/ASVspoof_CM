
from cv2 import log
from speechbrain.nnet.losses import nll_loss,bce_loss
import torch

def sasv_loss(
        asv_loss_metric,
        log_probabilities_asv,
        log_probabilities_cm,
        log_probabilities_tts,
        log_probabilities_vc,
        asv_targets,
        cm_targets,
        tts_targets,
        vc_targets,
        length=None,
        label_smoothing=0.0,
        allowed_len_diff=3,
        reduction="mean",
):
    # process cm loss
    cm_loss = bce_loss(log_probabilities_cm, cm_targets, length=length,reduction=reduction)

    if reduction == "batch":
        return cm_loss
    
    def get_masked_nll_loss(mask,
                            targets,
                            log_probabilities,
                            length,
                            loss_func
                            ):
        mask = torch.squeeze(mask, 1)
        masked_log_probabilities = log_probabilities[mask]
        masked_targets = targets[mask]
        masked_length = None
        # if length is not None: masked_length = length[torch.squeeze(mask, 1)]
        if length is not None: masked_length = length[mask]
        # if the target exists in current batch
        if masked_targets.shape[0] == 0:
            loss = torch.zeros(cm_loss.shape, device='cuda:0')
            return loss
        loss = loss_func(masked_log_probabilities,
                         masked_targets,
                         length=masked_length,
                         # reduction=reduction
                         )


        if reduction == 'batch':
            temp = torch.zeros(cm_loss.shape, device='cuda:0')
            for i, (index, value) in enumerate(zip(mask, loss)):
                temp[index] =value
            loss = temp


        return loss

    # process tts loss
    tts_mask = torch.le(tts_targets, 3)
    tts_loss = get_masked_nll_loss(tts_mask, tts_targets, log_probabilities_tts, length, nll_loss)

    # process vc loss
    vc_mask = torch.le(vc_targets, 1)
    vc_loss = get_masked_nll_loss(vc_mask, vc_targets, log_probabilities_vc, length, nll_loss)

    # process asv loss
    bonafide_mask = torch.ge(cm_targets, 1)
    asv_loss = get_masked_nll_loss(bonafide_mask, asv_targets, log_probabilities_asv, length, asv_loss_metric)

    ad_loss = 0.01*tts_loss + 0.01*vc_loss + asv_loss + cm_loss

    return ad_loss 

