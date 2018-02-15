import pandas as pd
import numpy as np
from scipy.stats.mstats import gmean
import sys
labels = pd.read_csv("labels.txt",sep= ' ',header=None)

df1 = pd.read_csv("fmow_imagenet1k-resnext-101-cnn-only-all_8_simplecut_test.txt",header=None)
df2 = pd.read_csv("fmow_imagenet11k-place365ch-resnet-152-cnn-only-all_6_simplecut_test.txt",header=None)
df = (df1+df2)/2
df[63] = df[63].astype(int)
df.columns = list(labels[0].values)+['_id']
meta1 = pd.read_csv("fmow_imagenet11k-resnet-152-cnn-meta-baseline-all_7_baseline_test.txt",header=None)
meta2 = pd.read_csv("fmow_imagenet11k-resnet-152-cnn-meta-baseline-all_7_simplecut_test.txt",header=None)

meta = (meta1+meta2)/2
meta[63] = meta[63].astype(int)
meta.columns = list(labels[0].values)+['_id']
print(meta.columns)
df = pd.concat([df,meta])
merge_meta = pd.read_csv("fmow_imagenet11k-resnet-152-cnn-meta-simplecut-all_7_simplecut_test.txt",header=None)
merge_meta.columns = list(labels[0].values)+['_id']
df = pd.concat([df,merge_meta])
base = pd.read_csv("../baseline/code/baseline_cnn_299_probs.csv")
df = pd.concat([df,base])
print(df.columns)
ids = list(set(df['_id'].values))
res = []
prob_res = []
def weight_probs(probs):
    cols = int(probs.shape[0]/4)
    weights = [0.8,1.2,1.2,0.8]
    #weights = [1,1,1,1]
    for idx in range(4):
        probs[idx*cols:idx*cols+cols] = probs[idx*cols:idx*cols+cols] * weights[idx]
    return probs
for _id in ids:
    probs = df[df['_id']==_id].values[:,1:]
    probs = weight_probs(probs)
    probs_mean = gmean(probs,axis=0) + np.mean(probs,axis=0)*0.5
    index =  np.argmax(probs_mean)
    prob_res.append(np.max(probs_mean,axis=0))
    res.append(df.columns[index+1])
df = pd.DataFrame({'_id':ids,'res':res})
df[['_id','res']].to_csv(sys.argv[1],header=None,index=False)
