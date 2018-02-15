# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

folder = "./data/simplecut_299/"

df = pd.read_csv("data/train_baseline.lst",header=None,sep='\t')
files = list(df[2].values)
files = map(lambda x:folder+x.replace('.jpg','.csv'),files)
print(len(files))
features = []#open(files[0], “r”).readline().split(",")
for fi in files[:]:
    data = open(fi, "r").readline().rstrip('\n').split(",")
    data = map(lambda x:float(x), data)
    features.append(data)
df = pd.DataFrame(np.array(features))
df.to_csv("train_280_features.csv",index=None,header=None)
