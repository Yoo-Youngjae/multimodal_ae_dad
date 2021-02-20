import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from modules.utils import Standardizer, Rotater, Truncater
import torch

def get_norm(x, norm_type=2):
    return abs(x)**norm_type

def get_auc_roc(score, test_label, writer=None, epoch=None, mode=None):
    try:
        fprs, tprs, threshold = metrics.roc_curve(test_label, score)
        print(mode,'auroc', metrics.auc(fprs, tprs))
        if writer is not None:
            fig = plt.figure()
            plt.title('Receiver Operating Characteristic')
            plt.plot(fprs, tprs, 'b', label='AUC = %0.2f' % metrics.auc(fprs, tprs))
            plt.legend(loc='lower right')
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            writer.add_figure("Performance/roc_curve_"+mode, fig, epoch)
            writer.add_text("ROC_curve/fprs" + mode, ' '.join([str(i) for i in fprs]), epoch)
            writer.add_text("ROC_curve/tprs" + mode, ' '.join([str(i) for i in tprs]), epoch)
        return metrics.auc(fprs, tprs)
    except:
        return .0

def get_auc_prc(score, test_label, writer=None, epoch=None, mode=None):
    try:
        precisions, recalls, threshold = metrics.precision_recall_curve(test_label, score)
        if writer is not None:
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
            writer.add_figure("Performance/precision-recall_curve_"+mode, fig, epoch)
        return metrics.auc(recalls, precisions)
    except:
        return .0

def get_f1_score(valid_score, test_score, test_label, f1_quantiles):


    precisions, recalls, thresholds = metrics.precision_recall_curve(test_label, test_score)
    f1_scores = 2 * recalls * precisions / (recalls + precisions)
    threshold = thresholds[np.argmax(f1_scores)]
    f1s = np.max(f1_scores)

    # threshold = np.quantile(valid_score, f1_quantiles)
    # predictions = test_score > threshold
    # p = (predictions & test_label).sum() / float(predictions.sum())
    # r = (predictions & test_label).sum() / float(sum(test_label)) #test_label.sum()
    #
    # f1s = p * r * 2 / (p + r)


    return f1s, threshold

def get_confusion_matrix(score, test_label, threshold,  writer=None, epoch=None, mode=None):
    score_label = []
    for i in score:
        if i >= threshold:
            score_label.append(True)   # positive
        else:
            score_label.append(False)   # negative

    tn, fp, fn, tp = metrics.confusion_matrix(test_label, score_label).ravel()
    conf_matrix_str = 'Tn, Fp : '+str(tn)+', '+str(fp)+' / Fn, Tp : '+str(fn)+', '+str(tp)
    # print('Tn, Fp : '+str(tn)+', '+str(fp)+'\nFn, Tp : '+str(fn)+', '+str(tp))
    if writer is not None:
        writer.add_text("Performance/confusion_matrix_"+mode, conf_matrix_str, epoch)
    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    return precision, recall


def get_recon_loss(test_diffs, valid_diffs, labels, writer, epoch, f1_quantiles=[.80], mode='base'):

    valid_losses = (valid_diffs ** 2).mean(axis=1)
    test_losses = (test_diffs ** 2).mean(axis=1)



    loss_auc_roc = get_auc_roc(test_losses, labels, writer, epoch, mode=mode)
    loss_auc_prc = get_auc_prc(test_losses, labels, writer, epoch, mode=mode)
    loss_f1s, threshold = get_f1_score( valid_losses,
                                        test_losses,
                                        labels,
                                        f1_quantiles=f1_quantiles)
    precision, recall = get_confusion_matrix(test_losses, labels, threshold, writer, epoch, mode=mode)
    return loss_auc_roc, loss_auc_prc, loss_f1s, precision, recall

def get_sap_loss(valid_diffs,
                test_diffs,
                test_label,
                writer,
                epoch,
                start_layer_index=0,
                end_layer_index=7,
                f1_quantiles=[.80],
                mode='sap'
                ):


    valid_diffs = torch.cat([torch.from_numpy(i) for i in valid_diffs[start_layer_index:end_layer_index]], dim=-1).numpy()
    test_diffs = torch.cat([torch.from_numpy(i) for i in test_diffs[start_layer_index:end_layer_index]], dim=-1).numpy()
    # |test_diffs| = (batch_size, dim * config.n_layers)

    d_loss = (test_diffs**2).mean(axis=1)
    d_loss_auc_roc = get_auc_roc(d_loss, test_label, writer, epoch, mode=mode)
    d_loss_auc_prc = get_auc_prc(d_loss, test_label, writer, epoch, mode=mode)
    d_loss_f1s, threshold = get_f1_score((valid_diffs**2).mean(axis=1),
                              d_loss,
                              test_label,
                              f1_quantiles=f1_quantiles
                             )
    precision, recall = get_confusion_matrix(d_loss, test_label, threshold, writer, epoch, mode=mode)
    return d_loss, d_loss_auc_roc, d_loss_auc_prc, d_loss_f1s, precision, recall

def get_nap_loss(train_diffs,
                    valid_diffs,
                    test_diffs,
                    test_label,
                    writer,
                    epoch,
                    f1_quantiles=[.80],
                    start_layer_index=0,
                    end_layer_index=7,
                    gpu_id=-1,
                    norm_type=2,
                    mode='nap'
                   ):


    train_diffs = torch.cat([torch.from_numpy(i) for i in train_diffs[start_layer_index:end_layer_index]], dim=-1).numpy()
    valid_diffs = torch.cat([torch.from_numpy(i) for i in valid_diffs[start_layer_index:end_layer_index]], dim=-1).numpy()

    test_diffs = torch.cat([torch.from_numpy(i) for i in test_diffs[start_layer_index:end_layer_index]], dim=-1).numpy()
    # |test_diffs| = (batch_size, dim * config.n_layers)

    rotater = Rotater()
    stndzer = Standardizer()

    rotater.fit(train_diffs, gpu_id=gpu_id)
    stndzer.fit(rotater.run(train_diffs, gpu_id=gpu_id))

    valid_rotateds = stndzer.run(rotater.run(valid_diffs, gpu_id=gpu_id))

    test_rotateds = stndzer.run(rotater.run(test_diffs, gpu_id=gpu_id))
    score = get_norm(test_rotateds, norm_type).mean(axis=1)
    auc_roc = get_auc_roc(score, test_label, writer, epoch, mode=mode)


    auc_prc = get_auc_prc(score, test_label, writer, epoch, mode=mode)
    try:
        f1_scores, threshold = get_f1_score(get_norm(valid_rotateds, norm_type).mean(axis=1),
                                 score,
                                 test_label,
                                 f1_quantiles=f1_quantiles
                                )
        precision, recall = get_confusion_matrix(score, test_label, threshold, writer, epoch, mode=mode)
    except Exception as e:
        print(e)
        f1_scores = 0
        precision, recall = 0

    return score, auc_roc, auc_prc, f1_scores, precision, recall
