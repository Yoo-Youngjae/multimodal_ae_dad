
import numpy as np
from sklearn import metrics
import torch
import matplotlib.pyplot as plt

def get_auc_roc(score, test_label, nap=False):
    try:
        fprs, tprs, threshold = metrics.roc_curve(test_label, score)
        return metrics.auc(fprs, tprs)
    except:
        return .0

def get_auc_prc(score, test_label):
    try:
        precisions, recalls, threshold = metrics.precision_recall_curve(test_label, score)
        # threshold = get_threshold(precisions, recalls, threshold)
        # precision, recall = get_confusion_matrix(score, test_label, threshold)
        show = False
        if show:
            pr_auc = metrics.auc(recalls, precisions)
            plt.title('Precision Recall Characteristic')
            plt.plot(recalls, precisions, 'b', label='AUC = %0.4f' % pr_auc)
            plt.legend(loc='lower right')
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('precisions')
            plt.xlabel('recalls')
            plt.show()
        return metrics.auc(recalls, precisions)
    except:
        return .0

def get_f1_score(valid_score, test_score, test_label, f1_quantiles=[.90]):

    threshold = np.quantile(valid_score, f1_quantiles)
    predictions = test_score > threshold
    p = (predictions & test_label).sum() / float(predictions.sum())
    r = (predictions & test_label).sum() / float(sum(test_label)) #test_label.sum()

    f1s = p * r * 2 / (p + r)

    return f1s, threshold


def get_recon_loss(test_losses, val_losses, labels, f1_quantiles=[.90]):
    labels = [i[0].item() for i in labels]
    loss_auc_roc = get_auc_roc(test_losses, labels)
    print('AUROC',loss_auc_roc)
    loss_auc_prc = get_auc_prc(test_losses, labels)
    print('AUPRC', loss_auc_prc)
    loss_f1s, threshold = get_f1_score( val_losses,
                                        test_losses,
                                        labels,
                                        f1_quantiles=f1_quantiles)
    precision, recall = get_confusion_matrix(test_losses, labels, threshold)
    return loss_auc_roc, loss_auc_prc, loss_f1s, precision, recall


def get_confusion_matrix(score, test_label, threshold):
    score_label = []
    for i in score:
        if i >= threshold:
            score_label.append(False)
        else:
            score_label.append(True)

    tn, fp, fn, tp = metrics.confusion_matrix(test_label, score_label).ravel()
    precision = tp/ (tp+fp)
    recall = tp/ (tp+fn)
    return precision, recall