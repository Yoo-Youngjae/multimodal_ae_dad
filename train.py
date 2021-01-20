import argparse
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from torch import optim
from modules.data_loader import get_loaders
from modules.Multisensory_Fusion import Multisensory_Fusion
import pandas as pd
import numpy as np

def get_config():
    parser = argparse.ArgumentParser(description='PyTorch Multimodal Time-series LSTM VAE Model')

    parser.add_argument('--epochs', type=int, default=30, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=96, help='batch size')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--clip', type=float, default=10, help='gradient clipping')
    parser.add_argument('--device_id', type=int, default=1, help='device id(default : 0)')

    parser.add_argument('--shuffle_data', action='store_true', default=True)
    parser.add_argument('--workers', type=int, default=4, help='number of workers')
    parser.add_argument('--seq_len', type=int, default=3, help='sequence length')
    parser.add_argument('--n_features', type=int, default=64, help='number of features')
    parser.add_argument('--embedding_dim', type=int, default=16, help='embedding dimension')


    parser.add_argument('--object_select_mode', action='store_true', default=False)
    parser.add_argument('--object_type', type=str, default="bottle")

    parser.add_argument('--sensor', type=str, default="force_torque") # All

    parser.add_argument('--origin_datafile_path', type=str, default="/data_ssd/hsr_dropobject/data/")
    parser.add_argument('--dataset_file_path', type=str, default="dataset/data_sum")
    parser.add_argument('--dataset_file_name', type=str, default="data_sum")

    parser.add_argument('--save_model_name', type=str, default="save/saveModel/result.pt")
    parser.add_argument('--save_data_name', type=str, default="dataset/data_sum.pt")
    parser.add_argument('--saved_result_csv_name', type=str, default="save/result_csv/result.csv")

    args = parser.parse_args()

    return args


def train(model, args, train_loader, valid_loader, epoch):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss().to(args.device_id)

    multisensory_fusion = Multisensory_Fusion(args)
    for epoch in range(args.batch_size):
        model.train()
        train_losses = []
        for r, d, m, t in train_loader:
            try:
                optimizer.zero_grad()
                train_input_representation = multisensory_fusion.fwd(r, d, m, t)
                train_input_representation = train_input_representation.to(args.device_id)
                train_output = model(train_input_representation)
                loss = criterion(train_output, train_input_representation)
                loss.backward()

                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                train_losses.append(loss.item())
            except Exception as e:
                pass

        val_losses = []
        model.eval()
        with torch.no_grad():
            for r, d, m, t in valid_loader:
                try:
                    valid_input_representation = multisensory_fusion.fwd(r, d, m, t)
                    valid_input_representation = valid_input_representation.to(args.device_id)
                    valid_output = model(valid_input_representation)
                    loss = criterion(valid_output, valid_input_representation)
                    val_losses.append(loss.item())
                except Exception as e:
                    pass
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')

    return model.eval()


def evaluate(model, args, test_loader, valid_loader, result_save=False):
    model.eval()
    predictions, losses = [], []
    criterion = nn.MSELoss().to(args.device_id)
    multisensory_fusion = Multisensory_Fusion(args)
    with torch.no_grad():
        for r, d, m, t in test_loader:
            test_input_representation = multisensory_fusion.fwd(r, d, m, t)
            test_input_representation = test_input_representation.to(args.device_id)
            test_output = model(test_input_representation)
            loss = criterion(test_output, test_input_representation)
            predictions.append(test_output.cpu().numpy().flatten())
            losses.append(loss.item())
        if result_save:
            df = pd.DataFrame([{'base_auroc': 0, 'sap_auroc': 0, 'base_f1score': 0, 'sap_f1score' : 0}])
            return df
    print(f'test loss {np.mean(losses)}')


if __name__ == '__main__':
    from model import model
    args = get_config()
    train_loader, valid_loader, test_loader = get_loaders(args)
    seq_len = args.seq_len
    n_features = args.n_features
    model = model.LSTM_AE(args, seq_len, n_features, embedding_dim=16)
    model = model.to(args.device_id)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    # train
    # for epoch in range(args.epochs):
    train(model, args, train_loader, valid_loader, 0)
    evaluate(model, args, test_loader, valid_loader)

    # save eval
    torch.save(model.state_dict(), args.save_model_name)
    df_eval = pd.DataFrame([{'base_auroc': 0, 'sap_auroc': 0, 'base_f1score':0, 'sap_f1score':0}])
    for i in range(30):
        df = evaluate(model, args, test_loader, valid_loader, result_save=True)
        df_eval = df_eval.append(df, ignore_index=True)

    df_eval[1:].to_csv(args.saved_result_csv_name)
