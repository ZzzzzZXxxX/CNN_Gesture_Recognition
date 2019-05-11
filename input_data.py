import os
import numpy as np
import glob
from skimage import io,transform
import pickle

# ============================================================================
# -----------------生成图片路径和标签的List------------------------------------
# step1：获取所有的图片路径名，存放到
# 对应的列表中，同时贴上标签，存放到label列表中。
def read_img(file_dir):
    # 将所有的图片resize成100*100
    w = 100
    h = 100
    c = 3
    catelist = os.listdir(file_dir)  # 获取改目录下所有子目录
    classes = []
    for i in catelist:
        if i == '.DS_Store':
            continue
        classes.append(i)
    all = []
    # step1：获取所有的图片路径名，存放到
    # 对应的列表中，同时贴上标签，存放到label列表中。
    # 遍历主文件夹下所有的类别文件夹
    imgs = []
    labels = []
    dict_ =[]
    for index, name in enumerate(classes):
        path = file_dir + name + '/'
        # 获取所有该类别文件夹下所有的图片路径
        path_all = glob.glob(path + '*.jpg')
        dict_.append((index,name))
        for img_ in path_all:
            print('reading the images:%s' % (img_))
            img = io.imread(img_)
            img = transform.resize(img, (w, h, c))
            imgs.append(img)
            labels.append(index)
    with open('dict.pickle', 'wb') as file:
        pickle.dump(dict(dict_), file)
    return np.asarray(imgs,np.float32),np.asarray(labels,np.int32),len(classes)
# ============================================================================
# -----------------生成训练与测试集------------------------------------
def set_val(data,label,ratio):
    # 打乱顺序
    num_example = data.shape[0]
    arr = np.arange(num_example)
    np.random.shuffle(arr)
    data = data[arr]
    label = label[arr]

    # 将所有数据分为训练集和验证集
    s = np.int(num_example * ratio)
    x_train = data[:s]
    y_train = label[:s]
    x_val = data[s:]
    y_val = label[s:]
    return x_train,y_train,x_val,y_val