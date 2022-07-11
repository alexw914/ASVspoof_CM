import torch
from speechbrain.utils.metric_stats import MetricStats
import numpy as np
from utils.eval_asvspoof import evaluate_asvspoof19, evaluate_asvspoof21


def compute_det_curve(target_scores, nontarget_scores):

    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))  # false rejection rates
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))  # false acceptance rates
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))  # Thresholds are the sorted scores

    return frr, far, thresholds

def compute_eer(target_scores, nontarget_scores):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return float(eer), thresholds[min_index]

class BinaryMetricStats(MetricStats):
    """Tracks binary metrics, such as precision, recall, F1, EER, etc.
    """
    def __init__(self, eval_opt="2019LA",positive_label=1):
        self.clear()
        self.positive_label = positive_label
        self.eval_opt = eval_opt

    def clear(self):
        self.ids = []
        self.scores = []
        self.labels = []
        self.summary = {}
        self.top1 = []

    def append(self, ids, scores, labels):
        """Appends scores and labels to internal lists.

        Does not compute metrics until time of summary, since
        automatic thresholds (e.g., EER) need full set of scores.

        Arguments
        ---------
        ids : list
            The string ids for the samples

        """
        self.ids.extend(ids)
        self.scores.extend(scores.detach())
        self.labels.extend(labels.detach())


    def summarize(self, field=None, threshold=None, beta=1, eps=1e-8):
        """Compute statistics using a full set of scores.

        Full set of fields:
         - TP - True Positive
         - TN - True Negative
         - FP - False Positive
         - FN - False Negative
         - FAR - False Acceptance Rate
         - FRR - False Rejection Rate
         - DER - Detection Error Rate (EER if no threshold passed)
         - precision - Precision (positive predictive value)
         - recall - Recall (sensitivity)
         - F-score - Balance of precision and recall (equal if beta=1)
         - MCC - Matthews Correlation Coefficient

        Arguments
        ---------
        field : str
            A key for selecting a single statistic. If not provided,
            a dict with all statistics is returned.
        threshold : float
            If no threshold is provided, equal error rate is used.
        beta : float
            How much to weight precision vs recall in F-score. Default
            of 1. is equal weight, while higher values weight recall
            higher, and lower values weight precision higher.
        eps : float
            A small value to avoid dividing by zero.
        """

        if isinstance(self.scores, list):
            self.scores = torch.stack(self.scores)
            self.labels = torch.stack(self.labels)

        if len(self.labels.shape)>1:
            self.labels = torch.squeeze(self.labels, 1)
        if threshold is None:
            # if self.positive_label == 1:
            #     positive_scores = torch.index_select(self.scores, 0,  torch.squeeze(self.labels.nonzero(), 1))
            #     # positive_scores = self.scores[self.labels.nonzero(as_tuple=True)]
            #     # negative_scores = self.scores[
            #     #     self.labels[self.labels == 0].nonzero(as_tuple=True)
            #     # ]
            #     negative_scores = torch.index_select(self.scores, 0, torch.squeeze((self.labels==0).nonzero(), 1))
            # else:
            #     negative_scores = torch.index_select(self.scores, 0,  torch.squeeze(self.labels.nonzero(), 1))
            #     positive_scores = torch.index_select(self.scores, 0, torch.squeeze((self.labels==0).nonzero(), 1))

            # positive_scores = self.scores[self.labels.nonzero(as_tuple=True)]
            # negative_scores = self.scores[
            #     self.labels[self.labels == 0].nonzero(as_tuple=True)
            # ]

            positive_scores = torch.index_select(self.scores, 0, torch.squeeze(self.labels.nonzero(), 1))
            negative_scores = torch.index_select(self.scores, 0, torch.squeeze((self.labels==0).nonzero(), 1))

            # eer, threshold = EER(positive_scores, negative_scores)

            # eer, threshold = EER(positive_scores, negative_scores)
            positive_scores = torch.squeeze(positive_scores)
            negative_scores = torch.squeeze(negative_scores)
            # eer, threshold = compute_eer(positive_scores.detach().cpu().numpy(),
            #                              negative_scores.detach().cpu().numpy()
            #                              )
            if self.eval_opt=="2019LA":
                eer, min_tDCF, min_tDCF_threshold = evaluate_asvspoof19(positive_scores.detach().cpu().numpy(),
                                                                        negative_scores.detach().cpu().numpy())
            elif self.eval_opt=="2021LA-progress":
                eer, min_tDCF, min_tDCF_threshold = evaluate_asvspoof21(positive_scores.detach().cpu().numpy(),
                                                                        negative_scores.detach().cpu().numpy(), phase="progress")
            elif self.eval_opt=="2021DF-progress":
                eer, threshold = compute_eer(positive_scores.detach().cpu().numpy(),
                                            negative_scores.detach().cpu().numpy())
            else:
                raise Exception

        # pred = (self.scores >= threshold).float()
        # true = self.labels

        self.summary['EER'] = eer
        if self.eval_opt=="2021LA-progress" or self.eval_opt=="2019LA": 
            self.summary['min_tDCF'] = min_tDCF
            self.summary['Threshold'] = min_tDCF_threshold
        # self.summary["AAC"] = sum(self.top1)/len(self.top1)
        # self.summary['TCDF'] = minDCF(positive_scores, negative_scores)

        # TP = self.summary["TP"] = float(pred.mul(true).sum())
        # TN = self.summary["TN"] = float((1.0 - pred).mul(1.0 - true).sum())
        # FP = self.summary["FP"] = float(pred.mul(1.0 - true).sum())
        # FN = self.summary["FN"] = float((1.0 - pred).mul(true).sum())
        #
        # self.summary["FAR"] = FP / (TP + TN + eps)
        # self.summary["FRR"] = FN / (TP + TN + eps)
        # self.summary["DER"] = (FP + FN) / (TP + TN + eps)
        #
        # self.summary["precision"] = TP / (TP + FP + eps)
        # self.summary["recall"] = TP / (TP + FN + eps)
        # self.summary["F-score"] = (
        #     (1.0 + beta ** 2.0)
        #     * TP
        #     / ((1.0 + beta ** 2.0) * TP + beta ** 2.0 * FN + FP)
        # )
        #
        # self.summary["MCC"] = (TP * TN - FP * FN) / (
        #     (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + eps
        # ) ** 0.5

        if field is not None:
            return self.summary[field]
        else:
            return self.summary


def EER(positive_scores, negative_scores):
    """Computes the EER (and its threshold).

    Arguments
    ---------
    positive_scores : torch.tensor
        The scores from entries of the same class.
    negative_scores : torch.tensor
        The scores from entries of different classes.

    Example
    -------
    >>> positive_scores = torch.tensor([0.6, 0.7, 0.8, 0.5])
    >>> negative_scores = torch.tensor([0.4, 0.3, 0.2, 0.1])
    >>> val_eer, threshold = EER(positive_scores, negative_scores)
    >>> val_eer
    0.0
    """

    # Computing candidate thresholds
    thresholds, _ = torch.sort(torch.cat([positive_scores, negative_scores]))
    thresholds = torch.unique(thresholds)

    # Adding intermediate thresholds
    interm_thresholds = (thresholds[0:-1] + thresholds[1:]) / 2
    thresholds, _ = torch.sort(torch.cat([thresholds, interm_thresholds]))

    # Computing False Rejection Rate (miss detection)
    positive_scores = torch.cat(
        len(thresholds) * [positive_scores.unsqueeze(0)]
    )
    pos_scores_threshold = positive_scores.transpose(0, 1) <= thresholds
    FRR = (pos_scores_threshold.sum(0)).float() / positive_scores.shape[1]
    del positive_scores
    del pos_scores_threshold

    # Computing False Acceptance Rate (false alarm)
    negative_scores = torch.cat(
        len(thresholds) * [negative_scores.unsqueeze(0)]
    )
    neg_scores_threshold = negative_scores.transpose(0, 1) > thresholds
    FAR = (neg_scores_threshold.sum(0)).float() / negative_scores.shape[1]
    del negative_scores
    del neg_scores_threshold

    # Finding the threshold for EER
    min_index = (FAR - FRR).abs().argmin()

    # It is possible that eer != fpr != fnr. We return (FAR  + FRR) / 2 as EER.
    EER = (FAR[min_index] + FRR[min_index]) / 2

    return float(EER), float(thresholds[min_index])



def minDCF(
    positive_scores, negative_scores, c_miss=1.0, c_fa=1.0, p_target=0.01
):
    """Computes the minDCF metric normally used to evaluate speaker verification
    systems. The min_DCF is the minimum of the following C_det function computed
    within the defined threshold range:

    C_det =  c_miss * p_miss * p_target + c_fa * p_fa * (1 -p_target)

    where p_miss is the missing probability and p_fa is the probability of having
    a false alarm.

    Arguments
    ---------
    positive_scores : torch.tensor
        The scores from entries of the same class.
    negative_scores : torch.tensor
        The scores from entries of different classes.
    c_miss : float
         Cost assigned to a missing error (default 1.0).
    c_fa : float
        Cost assigned to a false alarm (default 1.0).
    p_target: float
        Prior probability of having a target (default 0.01).


    Example
    -------
    >>> positive_scores = torch.tensor([0.6, 0.7, 0.8, 0.5])
    >>> negative_scores = torch.tensor([0.4, 0.3, 0.2, 0.1])
    >>> val_minDCF, threshold = minDCF(positive_scores, negative_scores)
    >>> val_minDCF
    0.0
    """

    # Computing candidate thresholds
    thresholds, _ = torch.sort(torch.cat([positive_scores, negative_scores]))
    thresholds = torch.unique(thresholds)

    # Adding intermediate thresholds
    interm_thresholds = (thresholds[0:-1] + thresholds[1:]) / 2
    thresholds, _ = torch.sort(torch.cat([thresholds, interm_thresholds]))

    # Computing False Rejection Rate (miss detection)
    positive_scores = torch.cat(
        len(thresholds) * [positive_scores.unsqueeze(0)]
    )
    pos_scores_threshold = positive_scores.transpose(0, 1) <= thresholds
    p_miss = (pos_scores_threshold.sum(0)).float() / positive_scores.shape[1]
    del positive_scores
    del pos_scores_threshold

    # Computing False Acceptance Rate (false alarm)
    negative_scores = torch.cat(
        len(thresholds) * [negative_scores.unsqueeze(0)]
    )
    neg_scores_threshold = negative_scores.transpose(0, 1) > thresholds
    p_fa = (neg_scores_threshold.sum(0)).float() / negative_scores.shape[1]
    del negative_scores
    del neg_scores_threshold

    c_det = c_miss * p_miss * p_target + c_fa * p_fa * (1 - p_target)
    c_min, min_index = torch.min(c_det, dim=0)

    return float(c_min), float(thresholds[min_index])
