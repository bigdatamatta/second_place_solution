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
import tqdm


def json_to_feature_vector(metadata_length, jsonData, bb):
    features = np.zeros(metadata_length, dtype=float)
    features[0] = float(jsonData['gsd'])
    x,y = utm_to_xy(jsonData['utm'])
    features[1] = x
    features[2] = y
    features[3] = float(jsonData['cloud_cover']) / 100.0
    date = dparser.parse(jsonData['timestamp'])
    features[4] = float(date.year)
    features[5] = float(date.month) / 12.0
    features[6] = float(date.day) / 31.0
    features[7] = float(date.hour) + float(date.minute)/60.0

    if jsonData['scan_direction'].lower() == 'forward':
        features[8] = 0.0
    else:
        features[8] = 1.0
    features[9] = float(jsonData['pan_resolution_dbl'])
    features[10] = float(jsonData['pan_resolution_start_dbl'])
    features[11] = float(jsonData['pan_resolution_end_dbl'])
    features[12] = float(jsonData['pan_resolution_min_dbl'])
    features[13] = float(jsonData['pan_resolution_max_dbl'])
    features[14] = float(jsonData['multi_resolution_dbl'])
    features[15] = float(jsonData['multi_resolution_min_dbl'])
    features[16] = float(jsonData['multi_resolution_max_dbl'])
    features[17] = float(jsonData['multi_resolution_start_dbl'])
    features[18] = float(jsonData['multi_resolution_end_dbl'])
    features[19] = float(jsonData['target_azimuth_dbl']) / 360.0
    features[20] = float(jsonData['target_azimuth_min_dbl']) / 360.0
    features[21] = float(jsonData['target_azimuth_max_dbl']) / 360.0
    features[22] = float(jsonData['target_azimuth_start_dbl']) / 360.0
    features[23] = float(jsonData['target_azimuth_end_dbl']) / 360.0
    features[24] = float(jsonData['sun_azimuth_dbl']) / 360.0
    features[25] = float(jsonData['sun_azimuth_min_dbl']) / 360.0
    features[26] = float(jsonData['sun_azimuth_max_dbl']) / 360.0
    features[27] = float(jsonData['sun_elevation_min_dbl']) / 90.0
    features[28] = float(jsonData['sun_elevation_dbl']) / 90.0
    features[29] = float(jsonData['sun_elevation_max_dbl']) / 90.0
    features[30] = float(jsonData['off_nadir_angle_dbl']) / 90.0
    features[31] = float(jsonData['off_nadir_angle_min_dbl']) / 90.0
    features[32] = float(jsonData['off_nadir_angle_max_dbl']) / 90.0
    features[33] = float(jsonData['off_nadir_angle_start_dbl']) / 90.0
    features[34] = float(jsonData['off_nadir_angle_end_dbl']) / 90.0
    features[35] = float(bb['box'][2])
    features[36] = float(bb['box'][3])
    features[37] = float(jsonData['img_width'])
    features[38] = float(jsonData['img_height'])
    features[39] = float(date.weekday())
    features[40] = min([features[35], features[36]]) / max([features[37], features[38]])
    features[41] = features[35] / features[37]
    features[42] = features[36] / features[38]
    features[43] = date.second
    if len(jsonData['bounding_boxes']) == 1:
        features[44] = 1.0
    else:
        features[44] = 0.0
    features[45] = float(bb['box'][0])
    features[46] = float(bb['box'][1])
    return features
                  
def utm_to_xy(zone):
    """
    Converts UTM zone to x,y values between 0 and 1.
    :param zone: UTM zone (string)
    :return (x,y): values between 0 and 1
    """
    nums = range(1,61);
    letters = string.ascii_lowercase[2:-2]
    if len(zone) == 2:
        num = int(zone[0:1])
    else:
        num = int(zone[0:2])
    letter = zone[-1].lower()
    numIndex = nums.index(num)
    letterIndex = letters.index(letter)
    x = float(numIndex) / float(len(nums)-1)
    y = float(letterIndex) / float(len(letters)-1)
    return (x,y)

def build_feature(json_file):

    file_name = json_file.split('/')[-1].split('.')[0]
    cls = json_file.split('/')[-3]#.split('.')[0]
    data = json.load(open(json_file))
    folder = data['bounding_boxes'][0]['category']
    box= data['bounding_boxes'][0]
    metadata_length = 47
    features = json_to_feature_vector(metadata_length, data, box)
    return file_name, cls, features 
def make_image(img,size):
    (height, width,_) = img.shape
    if height > size or width > size:
        if height > width:
            img = cv2.resize(img,(size, size), interpolation = cv2.INTER_AREA)
        else:
            img = cv2.resize(img,(size, size), interpolation = cv2.INTER_AREA)
    new_img = np.zeros((size,size,3))
    (height, width,_) = img.shape
    start_x = (size - width) / 2
    start_y = (size - height) / 2
    new_img[start_y:start_y+height,start_x:start_x+width] = img
    return new_img
def extract_box(file_dir,dataset,file_name,idx,folder,box,img,features):
    (height, width,_) = img.shape
    [x,y,w,h] = box
    def center(x,y,w,h,size):
        center_x = x + w / 2
        center_y = y + h / 2
        if w < size:
            new_x = center_x - size / 2
            new_w = size
        else:
            new_x = x
            new_w = w
        if h < size:
            new_y = center_y - size / 2
            new_h = size
        else:
            new_y = y
            new_h = h
        return new_x, new_y, new_w, new_h
    x,y,w,h = center(x,y,w,h,size)
    x1 = max(0,int(x))
    x2 = min(width,int(x+w))
    y1 = max(0,int(y))
    y2 = min(height,int(y+h))
    crop_img = img[y1:y2,x1:x2]
    file_dir_folder = os.path.join(file_dir,folder)
    new_image = os.path.join(file_dir_folder,"%s_%d_%s"%(dataset,idx,file_name))
    cv2.imwrite(new_image, make_image(crop_img,size=size))
    feature_file = new_image.replace('.jpg','.csv')
    df = pd.DataFrame(features.reshape(1,-1))
    df.to_csv(feature_file,index=False,header=None)
def crop_image(file_dir,dataset,json_file):
    rgb_file = json_file.replace('.json','.jpg')
    file_name = rgb_file.split('/')[-1]#.split('.')[0]
    data = json.load(open(json_file))
    img = cv2.imread(rgb_file)
    if isinstance(data['bounding_boxes'], dict):
        folder = data['bounding_boxes']['category']
        box= data['bounding_boxes']['box']
        metadata_length = 47
        features = json_to_feature_vector(metadata_length, data, data['bounding_boxes'])
        extract_box(file_dir,dataset,file_name,0,folder,box,img,features)    
    else:
        for idx in range(len(data['bounding_boxes'])):
            folder = data['bounding_boxes'][idx]['category']
            box= data['bounding_boxes'][idx]['box']
            metadata_length = 47
            features = json_to_feature_vector(metadata_length, data,data['bounding_boxes'][idx])            
            extract_box(file_dir,dataset,file_name,idx,folder,box,img,features)


datasets = ['train','val']
data_dir = sys.argv[1]# "../../data/"
data_dir = data_dir.replace('train','')
size = int(sys.argv[2])
for dataset in datasets:
    folders = glob.glob(data_dir+dataset+"/*") + [data_dir+dataset+"/false_detection"] 
    file_dir = './data/simplecut_%s'%size
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
    #exit()
    func = partial(crop_image,file_dir,dataset)
    p = Pool(15)
    r = list(tqdm.tqdm(p.imap(func, files), total=len(files)))
