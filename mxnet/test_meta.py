import os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
import sys
import numpy as np
import cv2
import json
from common import find_mxnet
import mxnet as mx
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
import imutils
def ch_dev(arg_params, aux_params, ctx):
    new_args = dict()
    new_auxs = dict()
    for k, v in arg_params.items():
        new_args[k] = v.as_in_context(ctx)
    for k, v in aux_params.items():
        new_auxs[k] = v.as_in_context(ctx)
    return new_args, new_auxs


#prefix = './model/imagenet11k-place365ch-resnet-152-cnn-meta-simplecut'
#epoch = 26#int(sys.argv[1]) #check point step
#prefix = './model/imagenet1k-resnext-101-dropout'
#prefix = sys.argv[1]
#epoch = 7 
prefix = sys.argv[1]
epoch = int(sys.argv[2])
size = int(sys.argv[3])
gpu_id = 0#int(sys.argv[2]) #GPU ID for infer
ctx = mx.gpu(1)
sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
arg_params, aux_params = ch_dev(arg_params, aux_params, ctx)

cnt = 0
img_sz = size 
crop_sz = size 
dataset = sys.argv[4]#'test'#sys.argv[2]#"val"

preds = []
im_idxs = []
batch_sz = 128 
input_blob = np.zeros((batch_sz,3,crop_sz,crop_sz))
#test_imgs = glob.glob('simplecut_%s_%s/*_rgb.jpg'%(dataset,img_sz))
test_imgs = glob.glob('./data/%s_%s/*_rgb.jpg'%(dataset,img_sz))
print(test_imgs[0])
test_imgs =sorted(test_imgs)
length_imgs = len(test_imgs)
#imgs_id = map(lambda x:x.split('/')[-1].split('-')[0],test_imgs)
imgs_id = map(lambda x:x.split('/')[-1].split('_')[1],test_imgs)
def load_image(tta,img_name):
    if tta == 0:
        shift_x, shift_y = 0, 0
    else:
        shift_x = np.random.randint(1,18)
        shift_y = np.random.randint(1,18) 
    img = cv2.imread(img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[shift_y:shift_y+crop_sz, shift_x:shift_x+crop_sz]#cv2.resize(img,(crop_sz,crop_sz))#img[10:10+crop_sz, 10:10+crop_sz]
    img = np.float32(img)
    img[:,:,2] -= 103.939
    img[:,:,1] -= 116.779
    img[:,:,0] -= 123.68
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    csv_name = img_name.replace('.jpg','.csv')
    csv_name = csv_name.replace('baseline','simplecut')
    meta = pd.read_csv(csv_name,header=None).values 
    return img,meta
p = Pool(batch_sz)
df = pd.read_csv("labels.txt",header=None,delimiter=' ')
label_dict = dict(zip(df[1].values,df[0].values))


def predict(tta,test_imgs,imgs_id,length_imgs, batch_sz):
    val_meta_data = pd.read_csv('train_280_features.csv',header=None)
    val_meta_data = val_meta_data.values
    val_meta_data = val_meta_data.max(axis=0)
    for i in tqdm(range(0, length_imgs, batch_sz)):
        chunk = test_imgs[i:i + batch_sz]
        imgs_chunk = imgs_id[i:i + batch_sz]
        input_blob = np.zeros((len(chunk),3,crop_sz,crop_sz))
        meta_blob = np.zeros((len(chunk),47))
        for idx,img_name in enumerate(chunk):
            input_blob[idx], meta_blob[idx] = load_image(tta,img_name)
        arg_params["data"] = mx.nd.array(input_blob, ctx)
        arg_params["meta_data"] = mx.nd.array(meta_blob/val_meta_data, ctx)
        arg_params["softmax_label"] = mx.nd.empty((len(input_blob),), ctx)
        exe = sym.bind(ctx, arg_params ,args_grad=None, grad_req="null", aux_states=aux_params)
        exe.forward(is_train=False)
        net_out = exe.outputs[0].asnumpy()
        sub = pd.DataFrame(net_out)
        sub['_id'] = imgs_chunk
        if i == 0: 
            sub.to_csv('fmow_%s_%d_%s.txt'%(prefix.split('/')[-1],epoch,dataset),index=False,header=None)
        else: 
            sub.to_csv('fmow_%s_%d_%s.txt'%(prefix.split('/')[-1],epoch,dataset),mode='a',index=False,header=None)

for tta in range(1):
    predict(tta,test_imgs,imgs_id,length_imgs, batch_sz)

