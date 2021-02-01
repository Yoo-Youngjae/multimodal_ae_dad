import pandas as pd
import librosa
import decimal
import numpy as np
from tqdm import tqdm
# source_path   path of audio file
# save_path     path of mfcc csv file to be saved
# length        desirable length of wav file (sec) -> 13
# window_size   It will extract features in 'window_size' seconds -> 0.1
# stride        It will extract features per 'stride' seconds -> 0.1
def save_mfcc_from_wav(source_path, save_path, length=decimal.Decimal(0.0), window_size=0.1, stride=0.1, start_time=0):
    # load wav
    print('go mfcc ')
    y, sr = librosa.load(source_path)

    # check if more than 'length' sec
    if len(y) < sr * length:
        print('length of wav file must be over ' + str(length) + ' seconds')

    # cut wav to exactly 'length' seconds
    length = round(length,1)
    slice_idx = round(sr*length)
    y = y[:slice_idx]

    # apply MFCC
    nfft = int(round(sr * window_size))
    hop = int(round(sr * stride))
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128, n_fft=nfft, hop_length=hop) #
    log_S = librosa.power_to_db(S, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=16)  # 13   # n_mfcc => number of features

    # reform [n_mfcc x length*(1/stride)] -> [length*(1/stride) x n_mfcc]
    mfcc = mfcc.T
    column_name = []
    # save results

    np.savetxt(save_path, mfcc, delimiter=',')

    for i in range(16):
        if i < 10:
            column_name.append('mfcc0' + str(i))
        else:
            column_name.append('mfcc' + str(i))
    df = pd.DataFrame(mfcc)
    df.columns = column_name
    return df

if __name__ == '__main__':
    root_data_dir = '/data_ssd/hsr_dropobject/data/'
    for i in range(1):
        df_datasum = pd.read_csv('../dataset/data_sum_free'+str(i)+'.csv')
        df = df_datasum[['id', 'cur_depth_id', 'cur_hand_weight', 'cur_hand_id', 'label', 'data_dir', 'now_timegap']]

        data_dir_series = df_datasum['data_dir']
        now_timegap_series = df_datasum['now_timegap']

        start_time = now_timegap_series[0]
        sound_df = None
        save_mode = False

        for idx, (timegap, data_dir) in tqdm(enumerate(zip(now_timegap_series, data_dir_series))):
            if idx == len(now_timegap_series) - 1:  # last row
                save_mode = True
            elif abs(timegap - now_timegap_series[idx+1]) > 0.11:   # last row
                save_mode = True
            if idx != len(now_timegap_series) - 1:
                print('start_time', start_time, 'gap :', round(abs(timegap - now_timegap_series[idx + 1]), 2), 'timegap', timegap, 'save_mode', save_mode)


            if save_mode:
                dir_name = root_data_dir + data_dir + '/data'
                length = timegap-start_time + 0.1
                if sound_df is None:
                    sound_df = save_mfcc_from_wav(dir_name + '/sound/output.wav', dir_name + '/sound/mfcc.csv', length=round(length, 2),
                                          window_size=0.1, stride=0.1, start_time=start_time)
                else:
                    temp_df = save_mfcc_from_wav(dir_name + '/sound/output.wav', dir_name + '/sound/mfcc.csv', length=round(length, 2),
                                                  window_size=0.1, stride=0.1, start_time=start_time)
                    sound_df = sound_df.append(temp_df, ignore_index=True)

                if idx != len(now_timegap_series) - 1: # last row
                    start_time = now_timegap_series[idx + 1]
                save_mode = False



        data_df = pd.concat([df, sound_df], axis=1)
        data_df.to_csv('../dataset/data_sum_free_new'+str(i)+'.csv')