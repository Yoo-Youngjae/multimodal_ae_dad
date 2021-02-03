import argparse
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from torch import optim
from modules.data_loader import get_loaders, get_n_features
from modules.Multisensory_Fusion import Multisensory_Fusion
from modules.evaluation_metric import get_recon_loss
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from tqdm import tqdm
from sklearn import metrics



def get_config():
    parser = argparse.ArgumentParser(description='PyTorch Multimodal Time-series LSTM VAE Model')

    parser.add_argument('--epochs', type=int, default=2, help='upper epoch limit') # 30
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size') # 64
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


    parser.add_argument('--sensor', type=str, default="hand_camera")  # All, force_torque,  mic, hand_camera

    parser.add_argument('--dataset_file_name', type=str, default="data_sum_free")   # data_sum, data_sum_free, data_sum_motion
    parser.add_argument('--log_memo', type=str, default="embedding_dim_128")



    args = parser.parse_args()

    return args


def train(model, args, train_loader, valid_loader, writer):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss().to(args.device_id)

    multisensory_fusion = Multisensory_Fusion(args)
    train_log_idx = 0
    valid_log_idx = 0
    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for r, d, m, t, label in tqdm(train_loader):
            try:
                optimizer.zero_grad()
                train_input_representation = multisensory_fusion.fwd(r, d, m, t)
                train_input_representation = train_input_representation.to(args.device_id)
                train_output = model(train_input_representation)
                loss = criterion(train_output, train_input_representation)
                writer.add_scalar("Train/train_loss", loss, train_log_idx)
                train_log_idx += 1
                loss.backward()

                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                train_losses.append(loss.item())
            except Exception as e:
                # print(e)
                continue

        val_losses = []
        model.eval()
        with torch.no_grad():
            for r, d, m, t, label in tqdm(valid_loader):
                try:
                    valid_input_representation = multisensory_fusion.fwd(r, d, m, t)
                    valid_input_representation = valid_input_representation.to(args.device_id)
                    valid_output = model(valid_input_representation)
                    loss = criterion(valid_output, valid_input_representation)
                    writer.add_scalar("Train/valid_loss", loss, valid_log_idx)
                    valid_log_idx += 1
                    val_losses.append(loss.item())
                except Exception as e:
                    pass

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')

    return model.eval()


def evaluate(model, args, test_loader, valid_loader, writer):
    model = model.to(args.device_id)
    args.batch_size = 1
    model.eval()
    predictions, losses = [], []
    criterion = nn.MSELoss().to(args.device_id)
    multisensory_fusion = Multisensory_Fusion(args)
    labels = []
    val_losses = []
    normal_losses = []
    abnormal_losses = []
    eval_valid_log_idx = 0
    eval_test_log_idx = 0

    with torch.no_grad():
        for r, d, m, t, label in tqdm(valid_loader):
            try:
                valid_input_representation = multisensory_fusion.fwd(r, d, m, t)
                valid_input_representation = valid_input_representation.to(args.device_id)
                valid_output = model(valid_input_representation)
                loss = criterion(valid_output, valid_input_representation)
                val_losses.append(loss.item())
            except Exception as e:
                pass

        for r, d, m, t, label in tqdm(test_loader):
            try:
                test_input_representation = multisensory_fusion.fwd(r, d, m, t)
                test_input_representation = test_input_representation.to(args.device_id)
                test_output = model(test_input_representation)
                loss = criterion(test_output, test_input_representation)
                predictions.append(test_output.cpu().numpy().flatten())
                losses.append(loss.item())
                labels.append(label)
                if label[0].item() == 0:
                    writer.add_scalar("Test/Normal_Loss", loss, eval_valid_log_idx)
                    eval_valid_log_idx += 1
                    normal_losses.append(loss.item())
                else:
                    writer.add_scalar("Test/Abnormal_Loss", loss, eval_test_log_idx)
                    eval_test_log_idx += 1
                    abnormal_losses.append(loss.item())

            except Exception as e:
                pass
        print(f'Mean normal_losses {np.mean(normal_losses)}')
        print(f'Mean abnormal_losses {np.mean(abnormal_losses)}')

        base_auroc, base_aupr, base_f1scores, base_precisions, base_recalls = get_recon_loss(losses, val_losses, labels, writer)
        print('base_f1scores, base_precisions, base_recalls', base_f1scores, base_precisions, base_recalls)
        return base_auroc, base_aupr, base_f1scores, base_precisions, base_recalls, np.mean(normal_losses), np.mean(abnormal_losses)



if __name__ == '__main__':
    # Below line is solution for [RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method]
    # torch.multiprocessing.set_start_method('spawn')
    from model import model

    args = get_config()
    # For tensorboard
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d-%H:%M:%S")
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

    # train
    train(model, args, train_loader, valid_loader, writer)
    base_auroc, base_aupr, base_f1scores, base_precisions, base_recalls, avg_normal_loss, avg_abnormal_loss = evaluate(model, args, test_loader, valid_loader, writer)

    writer.add_scalar("Performance/base_auroc", base_auroc, 0)
    writer.add_text("Performance/base_auroc", str(base_auroc))
    writer.add_text("Performance/base_aupr", str(base_aupr))
    writer.add_text("Performance/base_f1scores", str(base_f1scores))
    writer.add_text("Performance/base_precisions", str(base_precisions))
    writer.add_text("Performance/base_recalls", str(base_recalls))
    writer.add_text("Performance/avg_normal_loss", str(avg_normal_loss))
    writer.add_text("Performance/avg_abnormal_loss", str(avg_abnormal_loss))

    writer.close()

    # save eval
    # torch.save(model.state_dict(), 'save/saveModel/'+log_name+'.pt')

