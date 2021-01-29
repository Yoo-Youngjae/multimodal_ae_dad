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
        val = 1000 + 1000 + 64 + 64
        return val
    elif sensor == 'hand_camera':
        return 1000     # 1024
    elif sensor == 'head_depth':
        return 1000     # 512
    elif sensor == 'force_torque':
        return 64
    elif sensor == 'mic':
        return 64       # 128


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
        self.hand_camera = False
        self.head_depth = False
        self.compose = transforms.Compose([transforms.Resize((49, 49))])  # (224, 224)

        if args.sensor == 'All':
            self.All = True
            self.force_torque = True
            self.unimodal = False
            resnet50 = models.resnet50(pretrained=True)  # in (18, 34, 50, 101, 152)
            self.resnet50 = resnet50.to(self.args.device_id)
        elif args.sensor == 'force_torque':
            self.force_torque = True
        elif args.sensor == 'mic':
            self.mic = True
        elif args.sensor == 'hand_camera' or args.sensor == 'head_depth':
            if args.sensor == 'hand_camera':
                self.hand_camera = True
            elif args.sensor == 'head_depth':
                self.head_depth = True
            resnet50 = models.resnet50(pretrained=True)  # in (18, 34, 50, 101, 152)
            self.resnet50 = resnet50.to(self.args.device_id)



    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        idxs = self.idxs[idx]
        r = torch.tensor([])
        d = torch.tensor([])
        m = torch.tensor([])
        t = torch.tensor([])
        cur_rows = self.dataframe.loc[idxs[0]:idxs[2]]
        label = cur_rows['label'].tolist()

        # [ 0, 0, 1], [0, 1, 1], [1, 1, 1] is abnormal
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
        if self.mic:
            mic_df = cur_rows['mfcc00']
            for i in range(1, 13):
                if i < 10:
                    mic_df = pd.concat([mic_df, cur_rows['mfcc0' + str(i)]], axis=1)
                else:
                    mic_df = pd.concat([mic_df, cur_rows['mfcc' + str(i)]], axis=1)
            m = mic_df.to_numpy()
            # m = norm_vec_np(m)
            m = torch.from_numpy(m.astype(np.float32))
        if self.hand_camera:
            data_dirs = cur_rows['data_dir']

            img_dirs = cur_rows['cur_hand_id']
            sub_path = '/data/img/hand/'
            firstRow = True

            ## todo : Is this needed interpolation ?
            for img_dir, data_dir in zip(img_dirs, data_dirs):
                img_dir = self.args.origin_datafile_path + data_dir + sub_path + str(int(img_dir)) + '.png'
                im = Image.open(img_dir)
                im = self.compose(im)
                im_arr = np.array(im)

                if firstRow:
                    firstRow = False
                    base_im_arr = [im_arr]
                else:
                    base_im_arr = np.concatenate((base_im_arr, [im_arr]), axis=0)

            base_im_arr = base_im_arr.transpose((0, 3, 1, 2))
            im_arr = torch.FloatTensor(base_im_arr)
            im_arr = im_arr.to(self.args.device_id)
            with torch.no_grad():
                r = self.resnet50(im_arr)

        if self.head_depth:
            data_dirs = cur_rows['data_dir']
            img_dirs = cur_rows['cur_depth_id']
            sub_path = '/data/img/d/'

            firstRow = True

            ## todo : Is this needed interpolation ?
            for img_dir, data_dir in zip(img_dirs, data_dirs):
                img_dir = self.args.origin_datafile_path + data_dir + sub_path + str(int(img_dir)) + '.png'
                im = Image.open(img_dir)
                im = self.compose(im)
                im_arr = np.array(im)

                if firstRow:
                    firstRow = False
                    base_im_arr = [im_arr]
                else:
                    base_im_arr = np.concatenate((base_im_arr, [im_arr]), axis=0)

            base_im_arr = np.repeat(base_im_arr[..., np.newaxis], 3, -1)
            base_im_arr = base_im_arr.transpose((0, 3, 1, 2))
            im_arr = torch.FloatTensor(base_im_arr)
            im_arr = im_arr.to(self.args.device_id)
            with torch.no_grad():
                d = self.resnet50(im_arr)

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
    hand_camera = False
    head_depth = False

    if args.sensor == 'All':
        All = True
    elif args.sensor == 'force_torque':
        force_torque = True
    elif args.sensor == 'mic':
        mic = True
    elif args.sensor == 'hand_camera':
        hand_camera = True
    elif args.sensor == 'head_depth':
        head_depth = True

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
    elif hand_camera or head_depth:
        hand_series = df_datasum['cur_hand_id']
        depth_series = df_datasum['cur_depth_id']
        data_dir = df_datasum['data_dir']
        label_series = df_datasum['label']
        return pd.concat([hand_series, depth_series, data_dir, label_series], axis=1)

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






