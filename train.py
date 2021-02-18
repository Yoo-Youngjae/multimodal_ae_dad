import argparse
import torch
import torch.nn as nn
from torch import optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from tqdm import tqdm

from modules.data_loader import get_loaders, get_n_features, get_transformed_data
from modules.evaluation_metric import get_recon_loss, get_sap_loss, get_nap_loss
from modules.utils import get_diffs




def get_config():
    parser = argparse.ArgumentParser(description='PyTorch Multimodal Time-series LSTM VAE Model')

    parser.add_argument('--epochs', type=int, default=101, help='upper epoch limit') # 30
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--lr', type=float, default=0.0005, help='initial learning rate')
    parser.add_argument('--lr_alpha', type=float, metavar='M', default=0.0005,
                        help='initial learning rate (default: 5e-4)')
    parser.add_argument('--lr_beta', type=float, nargs='+', default=[0.9, 0.999],
                        help='exponential decay for momentum estimates (default: 0.9, 0.999)')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--clip', type=float, default=10, help='gradient clipping')
    parser.add_argument('--device_id', type=int, default=0, help='device id(default : 0)')
    parser.add_argument('--nap_device_id', type=int, default=1, help='device id(default : 0)')


    parser.add_argument('--seq_len', type=int, default=3, help='sequence length')
    parser.add_argument('--n_features', type=int, default=512, help='number of features')
    parser.add_argument('--n_layer', type=int, default=5, help='number of layer(encoder)')
    parser.add_argument('--origin_datafile_path', type=str, default="/data_ssd/hsr_dropobject/data/")
    parser.add_argument('--dataset_file_path', type=str, default="dataset/")
    parser.add_argument('--sensor', type=str, default="All")  # All, force_torque,  mic, hand_camera


    parser.add_argument('--object_select_mode', action='store_true', default=False) # True
    parser.add_argument('--object_type', type=str, default="book")          # cracker	doll	metalcup	eraser	cookies	book	plate	bottle
    parser.add_argument('--object_type_datafile_path', type=str, default="/data_ssd/hsr_dropobject/objectsplit.csv")
    parser.add_argument('--ae_type', type=str, default="ae")



    parser.add_argument('--embedding_dim', type=int, default=512, help='embedding dimension')  # 32, 128, 512
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')  # 64
    parser.add_argument('--shuffle_batch', action='store_true', default=True)
    parser.add_argument('--workers', type=int, default=8, help='number of workers')

    parser.add_argument('--dataset_file_name', type=str, default="data_sum")   # data_sum, data_sum_free, data_sum_motion
    parser.add_argument('--log_memo', type=str, default="ae_full")


    args = parser.parse_args()

    return args


def train(model, args, train_loader, writer, train_log_idx, valid_log_idx):
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=args.lr_alpha, betas=args.lr_beta, eps=1e-08,
                                 weight_decay=0, amsgrad=False)

    criterion = nn.MSELoss(reduction='sum').to(args.device_id)
    # criterion = nn.L1Loss().to(args.device_id)

    model.train()
    train_losses = []
    val_losses = []
    for r, d, m, t, label in tqdm(train_loader):
        try:
            optimizer.zero_grad()
            input_representation = model.fusion(r, d, m, t)
            train_output = model(input_representation)
            loss = criterion(train_output, input_representation)
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

    with torch.no_grad():
        for r, d, m, t, label in tqdm(valid_loader):
            try:
                input_representation = model.fusion(r, d, m, t)
                valid_output = model(input_representation)
                loss = criterion(valid_output, input_representation)
                writer.add_scalar("Train/valid_loss", loss, valid_log_idx)
                valid_log_idx += 1
                val_losses.append(loss.item())
            except Exception as e:
                pass


    return np.mean(train_losses), np.mean(val_losses), train_log_idx, valid_log_idx


def evaluate(epoch, model, args, test_loader, valid_loader, writer, valid_log_idx, eval_normal_log_idx, eval_abnormal_log_idx):
    model = model.to(args.device_id)
    model.eval()
    criterion = nn.MSELoss(reduction='sum').to(args.device_id)
    # criterion = nn.L1Loss().to(args.device_id)

    losses = []
    labels = []

    val_losses = []
    normal_losses = []
    abnormal_losses = []



    with torch.no_grad():
        for r, d, m, t, label in tqdm(valid_loader):
            try:
                input_representation = model.fusion(r, d, m, t)
                valid_output = model(input_representation)
                loss = criterion(valid_output, input_representation)
                writer.add_scalar("Train/valid_loss", loss, valid_log_idx)
                valid_log_idx += 1
                val_losses.append(loss.item())
            except Exception as e:
                pass
        eval_losses = []
        for r, d, m, t, label in tqdm(valid_loader):
            try:
                input_representation = model.fusion(r, d, m, t)
                valid_output = model(input_representation)
                for val_o, val_i, val_label in zip(valid_output, input_representation, label):
                    loss = criterion(val_o, val_i)
                    eval_losses.append(loss.item())
            except Exception as e:
                pass

        for r, d, m, t, label in tqdm(test_loader):
            try:
                input_representation = model.fusion(r, d, m, t)
                test_output = model(input_representation)
                for test_o, test_i, test_label in zip(test_output, input_representation, label):
                    loss = criterion(test_o, test_i)
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


        base_auroc, base_aupr, base_f1scores, base_precisions, base_recalls = get_recon_loss(losses, eval_losses, labels, writer, epoch)
        writer.add_scalar("Performance/base_auroc", base_auroc, epoch)
        writer.add_scalar("Performance/base_aupr", base_aupr, epoch)
        writer.add_scalar("Performance/base_f1scores", base_f1scores, epoch)
        writer.add_scalar("Performance/base_precisions", base_precisions, epoch)
        writer.add_scalar("Performance/base_recalls", base_recalls, epoch)
        writer.add_scalar("Performance/avg_normal_loss", np.mean(normal_losses), epoch)
        writer.add_scalar("Performance/avg_abnormal_loss", np.mean(abnormal_losses), epoch)
        return base_auroc, np.mean(val_losses), valid_log_idx, eval_normal_log_idx, eval_abnormal_log_idx

def test(model, args, train_loader, valid_loader, test_loader,  writer, epoch):
    with torch.no_grad():
        train_x, train_label = get_transformed_data(train_loader, model)
        valid_x, valid_label = get_transformed_data(valid_loader, model)
        test_x, _test_y = get_transformed_data(test_loader, model)

        train_diff_on_layers = get_diffs(args, train_x, model)
        valid_diff_on_layers = get_diffs(args, valid_x, model)
        test_diff_on_layers = get_diffs(args, test_x, model)

        _test_y = np.where(np.isin(_test_y, [1]), True, False).flatten()

        f1_quantiles=[.90]

        base_auroc, base_aupr, base_f1scores, base_precisions, base_recalls = get_recon_loss(
            test_diff_on_layers[0],
            valid_diff_on_layers[0],
            _test_y,
            writer,
            epoch,
            f1_quantiles=f1_quantiles
        )
        _, sap_auroc, sap_aupr, sap_f1scores, sap_precisions, sap_recalls = get_sap_loss(
            valid_diff_on_layers,
            test_diff_on_layers,
            _test_y,
            writer,
            epoch,
            f1_quantiles=f1_quantiles
        )

        score, nap_auroc, nap_aupr, nap_f1scores, nap_precisions, nap_recalls = get_nap_loss(
            train_diff_on_layers,
            valid_diff_on_layers,
            test_diff_on_layers,
            _test_y,
            writer,
            epoch,
            f1_quantiles=f1_quantiles,
            gpu_id=args.nap_device_id,
            norm_type=2,
        )

    writer.add_scalar("Performance/base_auroc", base_auroc, epoch)
    writer.add_scalar("Performance/base_aupr", base_aupr, epoch)
    writer.add_scalar("Performance/base_f1scores", base_f1scores, epoch)
    writer.add_scalar("Performance/base_precisions", base_precisions, epoch)
    writer.add_scalar("Performance/base_recalls", base_recalls, epoch)

    writer.add_scalar("Performance/sap_auroc", sap_auroc, epoch)
    writer.add_scalar("Performance/sap_aupr", sap_aupr, epoch)
    writer.add_scalar("Performance/sap_f1scores", sap_f1scores, epoch)
    writer.add_scalar("Performance/sap_precisions", sap_precisions, epoch)
    writer.add_scalar("Performance/sap_recalls", sap_recalls, epoch)

    writer.add_scalar("Performance/nap_auroc", nap_auroc, epoch)
    writer.add_scalar("Performance/nap_aupr", nap_aupr, epoch)
    writer.add_scalar("Performance/nap_f1scores", nap_f1scores, epoch)
    writer.add_scalar("Performance/nap_precisions", nap_precisions, epoch)
    writer.add_scalar("Performance/nap_recalls", nap_recalls, epoch)


    print('(base_auroc, base_aupr), (sap_auroc, sap_aupr), (nap_auroc, nap_aupr)',(base_auroc, base_aupr), (sap_auroc, sap_aupr), (nap_auroc, nap_aupr))
    return nap_auroc, nap_aupr, nap_f1scores



def set_best_model(args, val_loss, best_val_loss, best_model, model):
    from copy import deepcopy
    if best_val_loss is None:
        best_val_loss = val_loss
        best_model = deepcopy(model.state_dict())
    elif val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = deepcopy(model.state_dict())

    return best_val_loss, best_model


if __name__ == '__main__':
    # Below line is solution for [RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method]
    # torch.multiprocessing.set_start_method('spawn')
    from model import model
    from model.adversarial_auto_encoder import AdversarialAutoEncoder as AAE
    from model.variational_auto_encoder import VariationalAutoEncoder as VAE

    args = get_config()
    ## For tensorboard
    now = datetime.now()
    date_time = now.strftime("%m-%d-%H:%M:%S")
    log_name = date_time + '_' + args.sensor + '_' + args.dataset_file_name+'_'+args.log_memo
    logdir = 'log/' + log_name
    os.mkdir(logdir)
    writer = SummaryWriter(log_dir=logdir)
    print(log_name, 'start!')


    train_loader, valid_loader, test_loader = get_loaders(args)

    seq_len = args.seq_len
    args.n_features = get_n_features(args.sensor)
    n_features = args.n_features
    embedding_dim = args.embedding_dim

    if args.ae_type == 'lstm':
        model = model.LSTM_AE(args, seq_len, n_features, embedding_dim=embedding_dim)

    elif args.ae_type == 'vae':
            model = VAE(args,
                    input_size=n_features,
                    btl_size=embedding_dim,
                    n_layers=args.n_layer,
                    k=10
            )
    elif args.ae_type == 'aae':
        model = AAE(args,
                    input_size=n_features,
                    btl_size=embedding_dim,
                    n_layers=args.n_layer,
                )
    else:   # model_type == 'ae
        model = model.DNN_AE(args, seq_len, n_features, embedding_dim=embedding_dim)
    model = model.to(args.device_id)
    print(model)

    train_log_idx = 0
    valid_log_idx = 0
    eval_normal_log_idx = 0
    eval_abnormal_log_idx = 0
    best_model = None
    best_val_loss = None


    for epoch in range(args.epochs):
        # train
        train_loss, val_loss, train_log_idx, valid_log_idx = train(model, args, train_loader, writer, train_log_idx, valid_log_idx)

        # # set best model
        # best_val_loss, best_model = set_best_model(args, val_loss, best_val_loss, best_model, model)
        # model.load_state_dict(best_model)
        print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')

        # evaluation
        if epoch % 10 == 0:
            nap_auroc, nap_aupr, nap_f1scores = test(model, args, train_loader, valid_loader, test_loader,  writer, epoch//10)
            torch.cuda.empty_cache()

    sum_nap_auroc, sum_nap_aupr, sum_nap_f1scores = 0, 0, 0
    torch.save(model.state_dict(), 'save/saveModel/'+log_name+'.pt')
    for epoch in range(11, 41):
        nap_auroc, nap_aupr, nap_f1scores = test(model, args, train_loader, valid_loader, test_loader, writer, epoch)
        sum_nap_auroc += nap_auroc
        sum_nap_aupr += nap_aupr
        sum_nap_f1scores += nap_f1scores
        torch.cuda.empty_cache()

        # test
        # base_auroc, val_loss, valid_log_idx, eval_normal_log_idx, eval_abnormal_log_idx = evaluate(epoch,
        #     model, args, test_loader, valid_loader, writer, valid_log_idx, eval_normal_log_idx, eval_abnormal_log_idx)
    print('sum_nap_auroc, sum_nap_aupr, sum_nap_f1scores', sum_nap_auroc/30, sum_nap_aupr/30, sum_nap_f1scores/30)
    writer.add_scalar("Result/nap_auroc_avg", sum_nap_auroc/30, 0)
    writer.add_scalar("Result/nap_aupr_avg", sum_nap_aupr/30, 0)
    writer.add_scalar("Result/nap_f1scores_avg", sum_nap_f1scores/30, 0)
    writer.close()

    # save eval
    # torch.save(model.state_dict(), 'save/saveModel/'+log_name+'.pt')

