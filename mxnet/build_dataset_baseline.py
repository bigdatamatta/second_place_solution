import dateutil.parser as dparser
import numpy as np
import cv2
import glob
import os,sys
import json
from multiprocessing import Pool
from functools import partial
import string
import pandas as pd
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000
from keras.preprocessing import image
import tqdm
import warnings

warnings.simplefilter('ignore', Image.DecompressionBombWarning)
count = 0 
def make_image(file_dir,dataset,file_name,idx,folder,box,img,size):
    global count 
    # train with context around box
    contextMultWidth = 0.15
    contextMultHeight = 0.15

    wRatio = float(box[2]) / img.shape[0]
    hRatio = float(box[3]) / img.shape[1]

    if wRatio < 0.5 and wRatio >= 0.4:
        contextMultWidth = 0.2
    if wRatio < 0.4 and wRatio >= 0.3:
        contextMultWidth = 0.3
    if wRatio < 0.3 and wRatio >= 0.2:
        contextMultWidth = 0.5
    if wRatio < 0.2 and wRatio >= 0.1:
        contextMultWidth = 1
    if wRatio < 0.1:
        contextMultWidth = 2

    if hRatio < 0.5 and hRatio >= 0.4:
        contextMultHeight = 0.2
    if hRatio < 0.4 and hRatio >= 0.3:
        contextMultHeight = 0.3
    if hRatio < 0.3 and hRatio >= 0.2:
        contextMultHeight = 0.5
    if hRatio < 0.2 and hRatio >= 0.1:
        contextMultHeight = 1
    if hRatio < 0.1:
        contextMultHeight = 2


    widthBuffer = int((box[2] * contextMultWidth) / 2.0)
    heightBuffer = int((box[3] * contextMultHeight) / 2.0)

    r1 = box[1] - heightBuffer
    r2 = box[1] + box[3] + heightBuffer
    c1 = box[0] - widthBuffer
    c2 = box[0] + box[2] + widthBuffer

    if r1 < 0:
        r1 = 0
    if r2 > img.shape[0]:
        r2 = img.shape[0]
    if c1 < 0:
        c1 = 0
    if c2 > img.shape[1]:
        c2 = img.shape[1]

    if r1 >= r2 or c1 >= c2:
        pass
    else: 
        subImg = img[r1:r2, c1:c2, :]
        subImg = image.array_to_img(subImg)
        subImg = subImg.resize((size,size))
        file_dir_folder = os.path.join(file_dir,folder)
        imgPath = os.path.join(file_dir_folder,"%s_%d_%s"%(dataset,idx,file_name))
        subImg.save(imgPath)

    return


def crop_image(file_dir,dataset,json_file):
    rgb_file = json_file.replace('.json','.jpg')
    file_name = rgb_file.split('/')[-1]#.split('.')[0]
    data = json.load(open(json_file))
    img = image.load_img(rgb_file)
    img = image.img_to_array(img)
    if not isinstance(data['bounding_boxes'], list):
        data['bounding_boxes'] = [data['bounding_boxes']]

    for idx in range(len(data['bounding_boxes'])):
        folder = data['bounding_boxes'][idx]['category']
        box= data['bounding_boxes'][idx]['box']
        if box[2] <= 2 or box[3] <= 2:
            print(rgb_file)
            continue
        make_image(file_dir,dataset,file_name,idx,folder,box,img,size)

datasets = ['train','val']
data_dir = sys.argv[1]#"../../data/"
data_dir = data_dir.replace('train','')
size = int(sys.argv[2])
for dataset in datasets:
    folders = glob.glob(data_dir+dataset+"/*") + [data_dir+dataset+"/false_detection"] 
    file_dir = './data/baseline_%s'%size
    for folder in folders:
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)
        file_dir_folder = os.path.join(file_dir,folder.split('/')[-1])
        if not os.path.exists(file_dir_folder):
            os.mkdir(file_dir_folder)


    files = glob.glob(data_dir+dataset+"/*/*/*_rgb.json")
    print(len(files))
    #for json_file in tqdm.tqdm(files):
    #    crop_image(file_dir,dataset,json_file)
    func = partial(crop_image,file_dir,dataset)
    #with Pool(15) as p:
    p = Pool(15)    
    r = list(tqdm.tqdm(p.imap(func, files), total=len(files)))
    #p = Pool(15)
    #p.map(func,files)
