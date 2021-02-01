import pandas as pd
from matplotlib import pyplot as plt


df_datasum = pd.read_csv('../dataset/data_sum0.csv')
df = df_datasum[['id','mfcc00','mfcc01','mfcc02','mfcc03','mfcc04','mfcc05','mfcc06','mfcc07','mfcc08','mfcc09','mfcc10','mfcc11','mfcc12','label']]
idx = [i for i in range(len(df.index))]

for i in range(1,13):
    if i < 10:
        plt.plot(idx, df['mfcc0' + str(i)], label='mfcc0' + str(i))
    else:
        plt.plot(idx, df['mfcc' + str(i)], label='mfcc' + str(i))

plt.plot(idx, df['label'], label='label')
plt.legend()
plt.show()
