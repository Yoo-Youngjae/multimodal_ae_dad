
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
import os
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


def get_loaders(args):
    # 1. get whole dataset
    hsr_dataset = HSR_Dataset(args)
    # 2. split it in to train, valid, test.
    trainset, validset, testset = split_train_test(hsr_dataset)

    return DataLoader(trainset), DataLoader(validset), DataLoader(testset)


def HSR_Dataset(args):
    All = False
    force_torque = False
    unimodal = True

    if args.sensor == 'All':
        All = True
        force_torque = True
        unimodal = False
    elif args.sensor == 'force_torque':
        force_torque = True

    # 0. save file already existed
    if os.path.exists(args.save_data_name):
        data = torch.load(args.save_data_name)
        return data

    # 1. load csv files
    if args.dataset_file_name == 'data_sum':
        df_datasum = pd.read_csv(args.dataset_file_path + '0.csv')
        df_datasum = df_datasum.append(pd.read_csv(args.dataset_file_path + '1.csv'), ignore_index=True)
        df_datasum = df_datasum.append(pd.read_csv(args.dataset_file_path + '2.csv'), ignore_index=True)
        df_datasum = df_datasum.append(pd.read_csv(args.dataset_file_path + '3.csv'), ignore_index=True)
        df_datasum = df_datasum.append(pd.read_csv(args.dataset_file_path + '4.csv'), ignore_index=True)
        df_datasum = df_datasum.append(pd.read_csv(args.dataset_file_path + '5.csv'), ignore_index=True)
        df_datasum = df_datasum.append(pd.read_csv(args.dataset_file_path + '6.csv'), ignore_index=True)
        df_datasum = df_datasum.append(pd.read_csv(args.dataset_file_path + '7.csv'), ignore_index=True)
    else: # dataset_file_name == 'data_sum_motion' or 'data_sum_free'
        df_datasum = pd.read_csv(args.dataset_file_path + '0.csv')

    df_datasum.index = [i for i in range(len(df_datasum.index))]
    ## 2. essential erase
    if All:
        return df_datasum
    elif force_torque:
        hand_weight_series = df_datasum['cur_hand_weight']
        label_series = df_datasum['label']
        return pd.concat([hand_weight_series, label_series], axis=1)

    ##########################################################################


    hand_weight_series = df_datasum['cur_hand_weight']
    label_series = df_datasum['label']
    data_dir = df_datasum['data_dir']


    data = df_datasum.drop(columns=['data_dir'])
    data = data.drop(columns=['now_timegap'])
    data = data.drop(columns=['label'])
    data = data.drop(columns=['id'])
    # 2. LiDAR column already deleted in csv files
    # for i in range(963):
    #     if i < 10:
    #         data = data.drop(columns='LiDAR00' + str(i))
    #     elif i < 100:
    #         data = data.drop(columns='LiDAR0' + str(i))
    #     else:
    #         data = data.drop(columns='LiDAR' + str(i))
    data = data.loc[:, ~data.columns.str.match('Unnamed')]

    ## 3. sensor wised preprocess
    if All:
        # 1) sound data
        mic_df = None
        firstRow = True
        for i in range(13):
            if i == 0:
                mic_df = data['mfcc00']
            elif i < 10:
                mic_df = pd.concat([mic_df, data['mfcc0' + str(i)]], axis=1)
            else:
                mic_df = pd.concat([mic_df, data['mfcc' + str(i)]], axis=1)

        # 2) rgb, depth
        base_depth_arr = np.array([])
        base_hand_arr = np.array([])
        compose = transforms.Compose([transforms.Resize((112, 112))]) #(224, 224)
        for idx, data_dir_str in tqdm(zip(data.index, data_dir)):
            if idx == 100:
                break
            nowdf = data.loc[idx]

            hand_dir = args.origin_datafile_path + data_dir_str + '/data/img/hand/' + str(int(nowdf['cur_hand_id'])) + '.png'
            hand_im = Image.open(hand_dir)
            hand_im = compose(hand_im)
            hand_arr = np.array(hand_im)


            depth_dir = args.origin_datafile_path + data_dir_str + '/data/img/d/' + str(int(nowdf['cur_depth_id'])) + '.png'
            depth_im = Image.open(depth_dir)
            depth_im = compose(depth_im)
            depth_arr = np.array(depth_im)


            if firstRow:
                firstRow = False
                base_hand_arr = [hand_arr]
                base_depth_arr = [depth_arr]
            else:
                base_hand_arr = np.concatenate((base_hand_arr, [hand_arr]), axis=0)
                base_depth_arr = np.concatenate((base_depth_arr, [depth_arr]), axis=0)

        # transpose
        base_hand_arr = base_hand_arr.transpose((0, 3, 1, 2))
        base_depth_arr = np.repeat(base_depth_arr[..., np.newaxis], 3, -1)
        base_depth_arr = base_depth_arr.transpose((0, 3, 1, 2))

        torch.save(base_hand_arr, 'dataset/base_hand_arr7.pt')
        torch.save(base_depth_arr, 'dataset/base_depth_arr7.pt')

        base_hand_arr = torch.FloatTensor(base_hand_arr)
        base_hand_arr = base_hand_arr.to(args.device_id)
        base_depth_arr = torch.FloatTensor(base_depth_arr)
        base_depth_arr = base_depth_arr.to(1)

        resnet50 = models.resnet50(pretrained=True)  # in (18, 34, 50, 101, 152)
        resnet50_hand = resnet50.to(args.device_id)

        base_hand_arr = resnet50_hand(base_hand_arr)
        print(base_hand_arr.shape)
        resnet50_depth = resnet50.to(1)
        base_depth_arr = resnet50_depth(base_depth_arr)
        print(base_depth_arr.shape)



    # 4. multi modal fusion
        # 1) force_torque
    r = None
    d = None
    m = None
    t = None

    if force_torque:
        t = norm_vec_np(hand_weight_series.to_numpy())
        t = torch.from_numpy(t.astype(np.float32))
        t = t.view(-1, 1)
        print('t.shape', t.shape)
    if All:
        resnet50 = models.resnet50(pretrained=True) # in (18, 34, 50, 101, 152)
        # 2) hand_camera
        r = norm_vec_np(base_hand_arr)
        r = torch.from_numpy(r.astype(np.float32))
        r = r.view(-1, 1, 3, 24, 32).squeeze()
        r = F.interpolate(r, 32).view(-1, 1, 3, 32, 32)
        print('r.shape', r.shape)

        # 3) head_depth
        d = norm_vec_np(base_depth_arr)
        d = torch.from_numpy(d.astype(np.float32))
        d = d.view(-1, 1, 24, 32)
        d = F.interpolate(d, 32).view(-1, 1, 1, 32, 32)
        print('d.shape', d.shape)

        #4) mfcc
        mic_df = mic_df.to_numpy()
        m = norm_vec_np(mic_df)
        m = torch.from_numpy(m.astype(np.float32))
        m = m.view(-1, 1, 1, 13)
        print('m.shape', m.shape)
    multisensory_fusion = Multisensory_Fusion(unimodal,args)
    data = multisensory_fusion.forward(r,d,m,t)
    return data






def split_train_test(dataset):
    return None, None, None


def norm_vec_np(v, range_in=None, range_out=None):
    if range_out is None:
        range_out = [0.0,1.0]
    if range_in is None:
        range_in = [np.min(v,0), np.max(v,0)]
    r_out = range_out[1] - range_out[0]
    r_in = range_in[1] - range_in[0]
    v = (r_out * (v - range_in[0]) / r_in) + range_out[0]
    v = np.nan_to_num(v, nan=0.0)
    return v

