import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import random




def get_n_features(sensor):
    if sensor == 'All':
        val = 8 * 8 * 32
        return val
    elif sensor == 'hand_camera':
        return 512     # 1024
    elif sensor == 'head_depth':
        return 1000     # 512
    elif sensor == 'force_torque':
        return 256
    elif sensor == 'mic':
        return 256       # 128


class HsrDataset(Dataset):
    def __init__(self, args, idxs, dataframe, test=False):
        self.args = args
        self.idxs = idxs
        self.dataframe = dataframe

        self.batch_size = args.batch_size

        self.unimodal = True
        self.All = False
        self.force_torque = False
        self.mic = False
        self.hand_camera = False
        self.head_depth = False

        if args.sensor == 'All':
            self.All = True
            self.unimodal = False
        elif args.sensor == 'force_torque':
            self.force_torque = True
        elif args.sensor == 'mic':
            self.mic = True
        elif args.sensor == 'hand_camera':
                self.hand_camera = True
        elif args.sensor == 'head_depth':
            self.head_depth = True



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
        if 1 in label:  # 1 == positive
            label = 1
        else:           # 0 == negative
            label = 0

        if self.force_torque or self.All:
            hand_weight_series = cur_rows['cur_hand_weight']
            t = hand_weight_series.to_numpy()
            t = torch.from_numpy(t.astype(np.float32))    # t = torch.from_numpy(t.astype(np.float32))
        if self.mic or self.All:
            mic_df = cur_rows['mfcc00']
            for i in range(1, 16):
                if i < 10:
                    mic_df = pd.concat([mic_df, cur_rows['mfcc0' + str(i)]], axis=1)
                else:
                    mic_df = pd.concat([mic_df, cur_rows['mfcc' + str(i)]], axis=1)
            m = mic_df.to_numpy()
            m = torch.from_numpy(m.astype(np.float32))    # m = torch.from_numpy(m.astype(np.float32))
        if self.All:
            data_dirs = cur_rows['data_dir']
            r_img_dirs = cur_rows['cur_hand_id']
            d_img_dirs = cur_rows['cur_depth_id']
            r_sub_path = '/data/img/hand/'
            d_sub_path = '/data/img/d/'
            firstRow = True


            for r_img_dir, d_img_dir, data_dir in zip(r_img_dirs, d_img_dirs, data_dirs):
                r_img_dir = self.args.origin_datafile_path + data_dir + r_sub_path + str(int(r_img_dir)) + '.png'
                d_img_dir = self.args.origin_datafile_path + data_dir + d_sub_path + str(int(d_img_dir)) + '.png'

                r_im = Image.open(r_img_dir).resize((32, 32))
                r_im = np.array(r_im)
                d_im = Image.open(d_img_dir).resize((32, 32))
                d_im = np.array(d_im)
                d_im = d_im[:, :, np.newaxis]   # unsqueeze

                if firstRow:
                    firstRow = False
                    r_base_im_arr = [r_im]
                    d_base_im_arr = [d_im]


                else:
                    r_base_im_arr = np.concatenate((r_base_im_arr, [r_im]), axis=0)
                    d_base_im_arr = np.concatenate((d_base_im_arr, [d_im]), axis=0)

            r_base_im_arr = r_base_im_arr.transpose((0, 3, 1, 2))
            r = torch.FloatTensor(r_base_im_arr)

            d_base_im_arr = d_base_im_arr.transpose((0, 3, 1, 2))
            d = torch.FloatTensor(d_base_im_arr)


        if self.hand_camera:
            data_dirs = cur_rows['data_dir']
            r_img_dirs = cur_rows['cur_hand_id']
            r_sub_path = '/data/img/hand/'
            firstRow = True

            for r_img_dir, data_dir in zip(r_img_dirs, data_dirs):
                r_img_dir = self.args.origin_datafile_path + data_dir + r_sub_path + str(int(r_img_dir)) + '.png'

                r_im = Image.open(r_img_dir).resize((32, 32))
                r_im = np.array(r_im)

                if firstRow:
                    firstRow = False
                    r_base_im_arr = [r_im]


                else:
                    r_base_im_arr = np.concatenate((r_base_im_arr, [r_im]), axis=0)

            r_base_im_arr = r_base_im_arr.transpose((0, 3, 1, 2))
            r = torch.FloatTensor(r_base_im_arr)


        if self.head_depth:
            data_dirs = cur_rows['data_dir']
            d_img_dirs = cur_rows['cur_depth_id']
            d_sub_path = '/data/img/d/'
            firstRow = True

            for d_img_dir, data_dir in zip(d_img_dirs, data_dirs):
                d_img_dir = self.args.origin_datafile_path + data_dir + d_sub_path + str(int(d_img_dir)) + '.png'

                d_im = Image.open(d_img_dir).resize((32, 32))
                d_im = np.array(d_im)
                d_im = d_im[:, :, np.newaxis]  # unsqueeze

                if firstRow:
                    firstRow = False
                    d_base_im_arr = [d_im]
                else:
                    d_base_im_arr = np.concatenate((d_base_im_arr, [d_im]), axis=0)

            d_base_im_arr = d_base_im_arr.transpose((0, 3, 1, 2))
            d = torch.FloatTensor(d_base_im_arr)

        return r, d, m, t, label


def get_loaders(args):
    # 1. get whole dataset
    full_dataframe = get_Dataframe(args)

    # 2. split it in to train, valid, test.
    trainset, validset, testset = split_train_test(full_dataframe, args)

    return DataLoader(trainset, batch_size=args.batch_size, num_workers=args.workers, shuffle=args.shuffle_batch), \
           DataLoader(validset, batch_size=args.batch_size, num_workers=args.workers,shuffle=args.shuffle_batch), \
           DataLoader(testset, batch_size=args.batch_size, num_workers=args.workers)

def split_train_test(full_dataframe, args):
    # set to 0.3 seconds unit
    data_len = len(full_dataframe.index)
    # train 30800
    idxs = [[i, i+1, i+2] for i in range(data_len - 2)]
    normal_idx = []
    abnormal_idx = []
    normal_idx_dir = args.dataset_file_path + args.dataset_file_name + '_normal_idx.pt'        # _book_normal_idx.pt
    abnormal_idx_dir = args.dataset_file_path + args.dataset_file_name + '_abnormal_idx.pt'    # _book_abnormal_idx.pt
    if os.path.exists(normal_idx_dir):
        normal_idx = torch.load(normal_idx_dir)
        abnormal_idx = torch.load(abnormal_idx_dir)
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
        torch.save(normal_idx, normal_idx_dir)
        torch.save(abnormal_idx, abnormal_idx_dir)

    train_valid_test_ratio = [0.6, 0.2, 0.2]
    train_valid_size = [int(train_valid_test_ratio[0] * len(normal_idx)), int(train_valid_test_ratio[1] * len(normal_idx))]

    random.shuffle(normal_idx)

    trainset_idxs = normal_idx[:train_valid_size[0]]
    validset_idxs = normal_idx[train_valid_size[0]:train_valid_size[0] + train_valid_size[1]]
    testset_idxs = normal_idx[train_valid_size[0] + train_valid_size[1]:]

    # normal_idx = 50082
    # train 30049
    trainset = HsrDataset(args, trainset_idxs, full_dataframe)
    # valid 10016
    validset = HsrDataset(args, validset_idxs, full_dataframe)
    # test 10017 + 4800 = 14817
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
    # if os.path.exists(args.save_data_name):
    #     return torch.load(args.save_data_name)

    # 1. load csv files
    file_path = args.dataset_file_path + args.dataset_file_name     # dataset/data_sum
    if args.dataset_file_name == 'data_sum':
        df_datasum = pd.read_csv(file_path + '0.csv')
        df_datasum = df_datasum.append(pd.read_csv(file_path + '1.csv'), ignore_index=True)
        df_datasum = df_datasum.append(pd.read_csv(file_path + '2.csv'), ignore_index=True)
        df_datasum = df_datasum.append(pd.read_csv(file_path + '3.csv'), ignore_index=True)
        df_datasum = df_datasum.append(pd.read_csv(file_path + '4.csv'), ignore_index=True)
        df_datasum = df_datasum.append(pd.read_csv(file_path + '5.csv'), ignore_index=True)
        df_datasum = df_datasum.append(pd.read_csv(file_path + '6.csv'), ignore_index=True)
        df_datasum = df_datasum.append(pd.read_csv(file_path + '7.csv'), ignore_index=True)
        if args.object_select_mode:
            df_objectlist = pd.read_csv(args.object_type_datafile_path)
            print(args.object_type)
            df_objectlist = df_objectlist[args.object_type]
            object_dir_list = df_objectlist.to_list()
            df_datasum = df_datasum[df_datasum['data_dir'].isin(object_dir_list)]
    else:
        # dataset_file_name == 'data_sum_motion' or 'data_sum_free'
        df_datasum = pd.read_csv(file_path + '0.csv')

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
        for i in range(1, 16):
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
# for layer_wised diff
def get_transformed_data(data_loader, model):
    """
    Multi indexing support
    """
    x = []
    y = []
    for r, d, m, t, _y in tqdm(data_loader):
        try:
            _x = model.fusion(r, d, m, t)
            x.append(_x)
            y.append(_y)
        except Exception as e:
            pass

    if type(_x) == np.ndarray:
        x = np.stack(x)
    elif type(_x) == torch.Tensor:
        x = torch.stack(x)
    else:
        raise NotImplementedError

    if type(_y) == np.ndarray:
        y = np.array(y)
    elif type(_y) == torch.Tensor:
        y = torch.stack(y)

    return x, y



