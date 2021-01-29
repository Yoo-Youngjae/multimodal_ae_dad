import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
import os
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms



def get_n_features(sensor):
    if sensor == 'All':
        val = 1728
        return val
    elif sensor == 'hand_camera':
        return 1024
    elif sensor == 'force_torque':
        return 64
    elif sensor == 'head_depth':
        return 512
    elif sensor == 'LiDAR':
        return 2048
    elif sensor == 'mic':
        return 64


class HsrDataset(Dataset):
    def __init__(self, args, idxs, dataframe, test=False):
        self.args = args
        self.idxs = idxs
        self.dataframe = dataframe
        if test:
            self.batch_size = 1
        else:
            self.batch_size = args.batch_size
        self.unimodal = True
        self.All = False
        self.force_torque = False
        self.mic = False
        if args.sensor == 'All':
            self.All = True
            self.force_torque = True
            self.unimodal = False
        elif args.sensor == 'force_torque':
            self.force_torque = True
        elif args.sensor == 'mic':
            self.mic = True


    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        idxs = self.idxs[idx]
        r = torch.tensor([])
        d = torch.tensor([])
        m = torch.tensor([])
        t = torch.tensor([])
        cur_rows = self.dataframe.loc[idxs[0]:idxs[2]]
        # for i in idxs:
        label = cur_rows['label'].tolist()
        if 1 in label:
            label = 1
        else:
            label = 0
        if self.force_torque:
            hand_weight_series = cur_rows['cur_hand_weight']
            t = hand_weight_series.to_numpy()
            # t = norm_vec_np(t)
            t = torch.from_numpy(t.astype(np.float32))
            # t = t.view(-1, 1)
        elif self.mic:
            mic_df = self.dataframe.loc[idxs[0]:idxs[2]]['mfcc00']
            for i in range(1, 13):
                if i < 10:
                    mic_df = pd.concat([mic_df, cur_rows['mfcc0' + str(i)]], axis=1)
                else:
                    mic_df = pd.concat([mic_df, cur_rows['mfcc' + str(i)]], axis=1)
            m = mic_df.to_numpy()
            # m = norm_vec_np(m)
            m = torch.from_numpy(m.astype(np.float32))


        return r, d, m, t, label


def get_loaders(args):
    # 1. get whole dataset
    full_dataframe = get_Dataframe(args)

    # 2. split it in to train, valid, test.
    trainset, validset, testset = split_train_test(full_dataframe, args)

    return DataLoader(trainset, batch_size=args.batch_size, num_workers=args.workers, shuffle=args.shuffle_batch), \
           DataLoader(validset, batch_size=args.batch_size, num_workers=args.workers,shuffle=args.shuffle_batch), \
           DataLoader(testset, batch_size=1, num_workers=args.workers)

def split_train_test(full_dataframe, args):
    # set to 0.3 seconds unit
    data_len = len(full_dataframe.index)
    # train 30800
    idxs = [[i, i+1, i+2] for i in range(data_len - 2)]
    normal_idx = []
    abnormal_idx = []
    if os.path.exists('dataset/normal_idx.pt'):
        normal_idx = torch.load('dataset/normal_idx.pt')
        abnormal_idx = torch.load('dataset/abnormal_idx.pt')
    else:
        for a, b, c in tqdm(idxs):
            if full_dataframe.loc[a]['label'] == 0 and full_dataframe.loc[b]['label'] == 0 and full_dataframe.loc[c]['label'] == 0:
                normal_idx.append([a,b,c])
            elif (full_dataframe.loc[a]['label'] == 0 and full_dataframe.loc[b]['label'] == 0 and full_dataframe.loc[c]['label'] == 1):
                abnormal_idx.append([a, b, c])
                abnormal_idx.append([a+1, b+1, c+1])
                abnormal_idx.append([a+2, b+2, c+2])
                abnormal_idx.append([a+3, b+3, c+3])
                abnormal_idx.append([a+4, b+4, c+4])
        torch.save(normal_idx, 'dataset/normal_idx.pt')
        torch.save(abnormal_idx, 'dataset/abnormal_idx.pt')

    train_valid_test_ratio = [0.6, 0.2, 0.2]
    train_valid_size = [int(train_valid_test_ratio[0] * data_len), int(train_valid_test_ratio[1] * data_len)]

    trainset_idxs = normal_idx[:train_valid_size[0]]
    validset_idxs = normal_idx[train_valid_size[0]:train_valid_size[0] + train_valid_size[1]]
    testset_idxs = normal_idx[train_valid_size[0] + train_valid_size[1]:]


    # train 30800
    trainset = HsrDataset(args, trainset_idxs, full_dataframe)
    # valid 10200
    validset = HsrDataset(args, validset_idxs, full_dataframe)
    # test 15000
    testset = HsrDataset(args, testset_idxs+abnormal_idx, full_dataframe, test=True)
    return trainset, validset, testset


def get_Dataframe(args):
    All = False
    force_torque = False
    mic = False

    if args.sensor == 'All':
        All = True
        force_torque = True
    elif args.sensor == 'force_torque':
        force_torque = True
    elif args.sensor == 'mic':
        mic = True

    # 0. save file already existed
    if os.path.exists(args.save_data_name):
        return torch.load(args.save_data_name)

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
    else:
        # dataset_file_name == 'data_sum_motion' or 'data_sum_free'
        df_datasum = pd.read_csv(args.dataset_file_path + '0.csv')

    df_datasum.index = [i for i in range(len(df_datasum.index))]
    ## 2. essential erase
    if All:
        return df_datasum
    elif force_torque:
        hand_weight_series = df_datasum['cur_hand_weight']
        label_series = df_datasum['label']
        return pd.concat([hand_weight_series, label_series], axis=1)
    elif mic:
        mic_df = df_datasum['mfcc00']
        for i in range(1, 13):
            if i < 10:
                mic_df = pd.concat([mic_df, df_datasum['mfcc0' + str(i)]], axis=1)
            else:
                mic_df = pd.concat([mic_df, df_datasum['mfcc' + str(i)]], axis=1)
        label_series = df_datasum['label']
        return pd.concat([mic_df, label_series], axis=1)

    ##########################################################################

def norm_vec_np(v, range_in=None, range_out=None):
    if range_out is None:
        range_out = [0.0, 1.0]
    if range_in is None:
        range_in = [np.min(v,0), np.max(v,0)]
    r_out = range_out[1] - range_out[0]
    r_in = range_in[1] - range_in[0]
    v = (r_out * (v - range_in[0]) / r_in) + range_out[0]
    v = np.nan_to_num(v, nan=0.0)
    return v



def preprocess(self):
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






