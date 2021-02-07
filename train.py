import argparse
import torch
import torch.nn as nn
from torch import optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from tqdm import tqdm

from modules.data_loader import get_loaders, get_n_features
from modules.evaluation_metric import get_recon_loss




def get_config():
    parser = argparse.ArgumentParser(description='PyTorch Multimodal Time-series LSTM VAE Model')

    parser.add_argument('--epochs', type=int, default=100, help='upper epoch limit') # 30
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size') # 64
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--lr', type=float, default=0.0005, help='initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--clip', type=float, default=10, help='gradient clipping')
    parser.add_argument('--device_id', type=int, default=0, help='device id(default : 0)')

    parser.add_argument('--shuffle_batch', action='store_true', default=True)
    parser.add_argument('--workers', type=int, default=4, help='number of workers')
    parser.add_argument('--seq_len', type=int, default=3, help='sequence length')
    parser.add_argument('--n_features', type=int, default=512, help='number of features')
    parser.add_argument('--n_layer', type=int, default=5, help='number of layer(encoder)')
    parser.add_argument('--origin_datafile_path', type=str, default="/data_ssd/hsr_dropobject/data/")
    parser.add_argument('--dataset_file_path', type=str, default="dataset/")
    parser.add_argument('--object_select_mode', action='store_true', default=False)
    parser.add_argument('--object_type', type=str, default="bottle")
    parser.add_argument('--embedding_dim', type=int, default=128, help='embedding dimension')  # 32, 128

    parser.add_argument('--sensor', type=str, default="All")  # All, force_torque,  mic, hand_camera

    parser.add_argument('--dataset_file_name', type=str, default="data_sum")   # data_sum, data_sum_free, data_sum_motion
    parser.add_argument('--log_memo', type=str, default="batchnorm_3layer_at_MulFu")

    args = parser.parse_args()

    return args


def train(model, args, train_loader, writer, train_log_idx):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # criterion = nn.CrossEntropyLoss().to(args.device_id)
    criterion = nn.MSELoss().to(args.device_id)

    model.train()
    train_losses = []
    for r, d, m, t, label in tqdm(train_loader):
        try:
            optimizer.zero_grad()
            train_output, input_representation = model(r, d, m, t)
            loss = criterion(train_output, input_representation) ** 0.5
            writer.add_scalar("Train/train_loss", loss, train_log_idx)
            train_log_idx += 1
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            train_losses.append(loss.item())
        except Exception as e:
            print(e)
            continue


    return np.mean(train_losses), train_log_idx


def evaluate(epoch, model, args, test_loader, valid_loader, writer, valid_log_idx, eval_normal_log_idx, eval_abnormal_log_idx):
    model = model.to(args.device_id)
    model.eval()
    criterion = nn.MSELoss().to(args.device_id)
    # criterion = nn.CrossEntropyLoss().to(args.device_id)

    losses = []
    labels = []

    val_losses = []
    normal_losses = []
    abnormal_losses = []

    with torch.no_grad():
        for r, d, m, t, label in tqdm(valid_loader):
            try:
                valid_output, input_representation = model(r, d, m, t)
                loss = criterion(valid_output, input_representation) ** 0.5
                writer.add_scalar("Train/valid_loss", loss, valid_log_idx)
                valid_log_idx += 1
                val_losses.append(loss.item())
            except Exception as e:
                pass

        for r, d, m, t, label in tqdm(test_loader):
            try:
                test_output, input_representation = model(r, d, m, t)
                for test_o, test_i, test_label in zip(test_output, input_representation, label):
                    loss = criterion(test_o, test_i) ** 0.5
                    losses.append(loss.item())
                    labels.append(test_label)
                    if test_label.item() == 0:
                        writer.add_scalar("Test/Normal_Loss", loss, eval_normal_log_idx)
                        eval_normal_log_idx += 1
                        normal_losses.append(loss.item())
                    else:
                        writer.add_scalar("Test/Abnormal_Loss", loss, eval_abnormal_log_idx)
                        eval_abnormal_log_idx += 1
                        abnormal_losses.append(loss.item())
            except Exception as e:
                # print('test',e)
                pass

        base_auroc, base_aupr, base_f1scores, base_precisions, base_recalls = get_recon_loss(losses, val_losses, labels, writer, epoch)
        writer.add_scalar("Performance/base_auroc", base_auroc, epoch)
        writer.add_scalar("Performance/base_aupr", base_aupr, epoch)
        writer.add_scalar("Performance/base_f1scores", base_f1scores, epoch)
        writer.add_scalar("Performance/base_precisions", base_precisions, epoch)
        writer.add_scalar("Performance/base_recalls", base_recalls, epoch)
        writer.add_scalar("Performance/avg_normal_loss", np.mean(normal_losses), epoch)
        writer.add_scalar("Performance/avg_abnormal_loss", np.mean(abnormal_losses), epoch)
        return base_auroc, np.mean(val_losses), valid_log_idx, eval_normal_log_idx, eval_abnormal_log_idx



if __name__ == '__main__':
    # Below line is solution for [RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method]
    # torch.multiprocessing.set_start_method('spawn')
    from model import model

    args = get_config()
    ## For tensorboard
    now = datetime.now()
    date_time = now.strftime("%m-%d-%H:%M:%S")
    log_name = date_time + '_' + args.sensor + '_' + args.dataset_file_name+'_'+args.log_memo
    logdir = 'log/' + log_name
    os.mkdir(logdir)
    writer = SummaryWriter(log_dir=logdir)


    train_loader, valid_loader, test_loader = get_loaders(args)

    seq_len = args.seq_len
    args.n_features = get_n_features(args.sensor)
    n_features = args.n_features
    embedding_dim = args.embedding_dim

    model = model.LSTM_AE(args, seq_len, n_features, embedding_dim=embedding_dim)
    model = model.to(args.device_id)
    print(model)

    train_log_idx = 0
    valid_log_idx = 0
    eval_normal_log_idx = 0
    eval_abnormal_log_idx = 0

    for epoch in range(args.epochs):
        # train
        train_loss, train_log_idx = train(model, args, train_loader, writer, train_log_idx)
        # test
        base_auroc, val_loss, valid_log_idx, eval_normal_log_idx, eval_abnormal_log_idx = evaluate(epoch,
            model, args, test_loader, valid_loader, writer, valid_log_idx, eval_normal_log_idx, eval_abnormal_log_idx)
        print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss} base_auroc {base_auroc}')

    writer.close()

    # save eval
    # torch.save(model.state_dict(), 'save/saveModel/'+log_name+'.pt')

