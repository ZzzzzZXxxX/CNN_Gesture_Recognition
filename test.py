from skimage import io, transform
import tensorflow as tf
import numpy as np
import pickle
import glob
import os

path_all = glob.glob('test/' + '*.jpg')
with open('dict.pickle', 'rb') as file:
    dict_ = pickle.load(file)

w = 100
h = 100
c = 3


def read_label(name):
    a = ""
    for i in name:
        a = i
        break
    return a


def read_one_image(path):
    img = io.imread(path)
    img = transform.resize(img, (w, h))
    return np.asarray(img)


with tf.Session() as sess:
    data = []
    for path in path_all:
        data.append(read_one_image(path))
    saver = tf.train.import_meta_graph('save/model.meta')
    saver.restore(sess, tf.train.latest_checkpoint('save/'))

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    feed_dict = {x: data}

    logits = graph.get_tensor_by_name("logits_eval:0")

    classification_result = sess.run(logits, feed_dict)

    # 打印出预测矩阵
    # print(classification_result)
    # 打印出预测矩阵每一行最大值的索引
    # print(tf.argmax(classification_result, 1).eval())
    output = []
    output = tf.argmax(classification_result, 1).eval()
    error = 0
    for i, j in zip(path_all, range(len(output))):
        if read_label(os.path.basename(i)) != dict_[output[j]]:
            error = error + 1
        # print(i, "手势预测:" + dict_[output[j]])
print("检测目录：", os.path.dirname(i), "   检测图片数目：%d" % len(path_all), "   正确率：%f" % (1 - (error / len(path_all))))
