import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

def get_auc_roc(score, test_label, writer, epoch, nap=False):
    try:
        fprs, tprs, threshold = metrics.roc_curve(test_label, score)

        fig = plt.figure()
        plt.title('Receiver Operating Characteristic')
        plt.plot(fprs, tprs, 'b', label='AUC = %0.2f' % metrics.auc(fprs, tprs))
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        writer.add_figure("Performance/roc_curve", fig, epoch)
        return metrics.auc(fprs, tprs)
    except:
        return .0

def get_auc_prc(score, test_label, writer, epoch):
    try:
        precisions, recalls, threshold = metrics.precision_recall_curve(test_label, score)

        fig = plt.figure()
        pr_auc = metrics.auc(recalls, precisions)
        plt.title('Precision Recall Characteristic')
        plt.plot(recalls, precisions, 'b', label='AUC = %0.4f' % pr_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('precisions')
        plt.xlabel('recalls')
        writer.add_figure("Performance/precision-recall_curve", fig, epoch)
        return metrics.auc(recalls, precisions)
    except:
        return .0

def get_f1_score(valid_score, test_score, test_label, f1_quantiles=[.90]):

    threshold = np.quantile(valid_score, f1_quantiles)
    print('threshold', threshold)
    predictions = test_score > threshold
    p = (predictions & test_label).sum() / float(predictions.sum())
    r = (predictions & test_label).sum() / float(sum(test_label)) #test_label.sum()

    f1s = p * r * 2 / (p + r)

    return f1s, threshold


def get_recon_loss(test_losses, val_losses, labels, writer, epoch, f1_quantiles=[.93]):
    _labels = []
    for i in labels:
        if i.item() == 0:
            _labels.append(False)
        else:
            _labels.append(True)
    labels = _labels
    loss_auc_roc = get_auc_roc(test_losses, labels, writer, epoch)
    loss_auc_prc = get_auc_prc(test_losses, labels, writer, epoch)
    loss_f1s, threshold = get_f1_score( val_losses,
                                        test_losses,
                                        labels,
                                        f1_quantiles=f1_quantiles)
    precision, recall = get_confusion_matrix(test_losses, labels, threshold, writer, epoch)
    return loss_auc_roc, loss_auc_prc, loss_f1s, precision, recall


def get_confusion_matrix(score, test_label, threshold, writer, epoch):
    score_label = []
    for i in score:
        if i >= threshold:
            score_label.append(True)   # positive
        else:
            score_label.append(False)   # negative

    tn, fp, fn, tp = metrics.confusion_matrix(test_label, score_label).ravel()
    conf_matrix_str = 'Tn, Fp : '+str(tn)+', '+str(fp)+' / Fn, Tp : '+str(fn)+', '+str(tp)
    print('Tn, Fp : '+str(tn)+', '+str(fp)+'\nFn, Tp : '+str(fn)+', '+str(tp))
    writer.add_text("Performance/confusion_matrix", conf_matrix_str, epoch)
    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    return precision, recall