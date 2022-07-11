from typing import List, Union

import numpy, torch, json, os
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve


def get_all_EERs(
    preds: Union[torch.Tensor, List, numpy.ndarray], keys: List
) -> List[float]:
    """
    Calculate all three EERs used in the SASV Challenge 2022.
    preds and keys should be pre-calculated using dev or eval protocol in
    either 'protocols/ASVspoof2019.LA.asv.dev.gi.trl.txt' or
    'protocols/ASVspoof2019.LA.asv.eval.gi.trl.txt'

    :param preds: list of scores in tensor
    :param keys: list of keys where each element should be one of
    ['target', 'nontarget', 'spoof']
    """
    sasv_labels, sv_labels, spf_labels = [], [], []
    sv_preds, spf_preds = [], []

    for pred, key in zip(preds, keys):
        if key == "target":
            sasv_labels.append(1)
            sv_labels.append(1)
            spf_labels.append(1)
            sv_preds.append(pred)
            spf_preds.append(pred)

        elif key == "nontarget":
            sasv_labels.append(0)
            sv_labels.append(0)
            sv_preds.append(pred)

        elif key == "spoof":
            sasv_labels.append(0)
            spf_labels.append(0)
            spf_preds.append(pred)
        else:
            raise ValueError(
                f"should be one of 'target', 'nontarget', 'spoof', got:{key}"
            )

    fpr, tpr, _ = roc_curve(sasv_labels, preds, pos_label=1)
    sasv_eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)

    fpr, tpr, _ = roc_curve(sv_labels, sv_preds, pos_label=1)
    sv_eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)

    fpr, tpr, _ = roc_curve(spf_labels, spf_preds, pos_label=1)
    spf_eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)

    return sasv_eer, sv_eer, spf_eer


def get_verification_scores(test_data, enroll_data, test_dict, enrol_dict, cm_scores_dict):
    """ Computes positive and negative scores given the verification split.
    """
    asv_preds, cm_preds, keys, submissions = [], [], [], []

    OUTPUT_DIR = 'predictions'
    PREDS_FILE = 'sasv_preds.json'
    KEYS_FILE =  'sasv_keys.json'
    CM_PREDS_FILE = 'cm_preds.json'
    SUBMISSION_FILE = 'eval_submission.txt'

    # Cosine similarity initialization
    similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    for utt in test_data:
        speaker_id, utt_id, cm_id, target = utt.split()
        enrol_utt_list = enroll_data[speaker_id]
        test_emb = test_dict[utt_id]
        enrol_emb_list = [enrol_dict[enrol_id] for enrol_id in enrol_utt_list]
        scores = [similarity(test_emb, enrol_emb)[0] for enrol_emb in enrol_emb_list]
        cur_score = sum(scores)/len(scores)

        asv_preds.append(cur_score.cpu().data.numpy()[0].item())
        keys.append(target)
        if cm_scores_dict[utt_id] < -0.9:
            cur_score = 0
        submissions.append('%s %s %s %s %f'%(speaker_id,
                                                utt_id,
                                                cm_id,
                                                target,
                                                cur_score,
                                                ))

        cm_preds.append(cm_scores_dict[utt_id])

    with open(os.path.join(OUTPUT_DIR, PREDS_FILE),'w') as f:
        json.dump(asv_preds, f)
    with open(os.path.join(OUTPUT_DIR, KEYS_FILE), 'w') as f:
        json.dump(keys, f)
    with open(os.path.join(OUTPUT_DIR, CM_PREDS_FILE),'w') as f:
        json.dump(cm_preds, f)
    with open(os.path.join(OUTPUT_DIR, SUBMISSION_FILE), 'w') as f:
        for v in submissions:
            f.write(v+'\n')

    print('PREDICTION FINISHED!!!')
    preds = []
    for i in range(len(asv_preds)):
        if cm_preds[i] < - 0.9:
            preds.append(cm_preds[i])
        else:
            preds.append(asv_preds[i])
    sasv_eer, sv_eer, spf_eer=get_all_EERs(preds, keys)
    print("SASV EER: {:.2f} SV EER: {:.2f} SPF EER: {:.2f}".format(sasv_eer*100, sv_eer*100, spf_eer*100))