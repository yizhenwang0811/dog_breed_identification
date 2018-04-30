import tensorflow as tf

import vgg19_trainable as vgg19
import vgg19 as vgg19_test
import utils
import numpy as np
from skimage import io, transform,color
import glob
import os

import time
import csv
import scipy.io as sio
import xml.etree.ElementTree as ET
import cv2 as cv
from keras.preprocessing import image


def rotate(x, theta, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = image.transform_matrix_offset_center(rotation_matrix, h, w)
    x = image.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


def shift(x, wshift, hshift, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    h, w = x.shape[row_axis], x.shape[col_axis]  # 读取图片的高和宽
    tx = hshift * h  # 高偏移大小，若不偏移可设为0，若向上偏移设为正数
    ty = wshift * w  # 宽偏移大小，若不偏移可设为0，若向左偏移设为正数
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])
    transform_matrix = translation_matrix
    x = image.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


def zoom(x, zx, zy, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])
    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = image.transform_matrix_offset_center(zoom_matrix, h, w)  # 保持中心坐标不改变
    x = image.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


def shear(x, shear, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])
    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = image.transform_matrix_offset_center(shear_matrix, h, w)
    x = image.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        img = color.rgb2hsv(image)
        h, s, v = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])

        h = h + hue_shift

        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = s + sat_shift

        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = v + val_shift

        img[:, :, 0], img[:, :, 1], img[:, :, 2] = h, s, v

        image = color.hsv2rgb(img)

    return image


def random_channel_shift(x, intensity, channel_index=0):
    x = np.rollaxis(x, channel_index, 0)
    min_x, max_x = np.min(x), np.max(x)
    channel_images = [np.clip(x_channel + np.random.uniform(-intensity, intensity), min_x, max_x)
                      for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index + 1)
    return x

def get_bboxs(dirpath, annotation):
    tree = ET.parse(dirpath + annotation)
    root = tree.getroot()
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    lists = []
    for neighbor in root.iter('xmin'):
        xmin.append(neighbor.text)
    for neighbor in root.iter('ymin'):
        ymin.append(neighbor.text)
    for neighbor in root.iter('xmax'):
        xmax.append(neighbor.text)
    for neighbor in root.iter('ymax'):
        ymax.append(neighbor.text)
    lists.append(xmin)
    lists.append(ymin)
    lists.append(xmax)
    lists.append(ymax)
    lists = np.asarray(lists, np.int32)

    return lists


def read_train(annot_path, img_path, cell):
    w = 224
    h = 224
    images = cell[0]
    annotation = cell[1]
    labels = cell[2]
    imgs = []
    label = []
    bboxs = []
    for i in range(len(images)):
        print("Reading training image: " + images[i][0][0])
        img = cv.imread(img_path + images[i][0][0])
        bbox = get_bboxs(annot_path, annotation[i][0][0])
        #         print(bbox.shape)
        #         box = []
        for idx in range(bbox.shape[1]):
            imgp = img[bbox[1, idx]: bbox[3, idx], bbox[0, idx]: bbox[2, idx]]
            #             print(imgp.shape)
            if imgp.shape[0] == 0 or imgp.shape[1] == 0 or imgp.shape[2] != 3:
                print("ERROR")
                return 0
            imgp = transform.resize(imgp, (w, h), 1, 'constant')
            imgs.append(imgp)
            label.append(labels[i])
            for p in range(0):
                rotate_limit = (-20, 20)
                theta = np.pi / 180 * np.random.uniform(rotate_limit[0], rotate_limit[1])
                img_rot = rotate(imgp, theta)
                imgs.append(img_rot)
                label.append(labels[i])

                zoom_range = (0.8, 1)
                zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
                img_zoom = zoom(imgp, zx, zy)
                imgs.append(img_zoom)
                label.append(labels[i])

                intensity = 0.3
                sh = np.random.uniform(-intensity, intensity)
                img_shear = shear(imgp, sh)
                imgs.append(img_shear)
                label.append(labels[i])


    print(np.array(imgs).shape)
    return np.asarray(imgs, np.float32), np.asarray(label, np.int32), len(np.unique(label))


def read_test(annot_path, img_path, cell):
    w = 224
    h = 224
    images = cell[0]
    annotation = cell[1]
    labels = cell[2]
    imgs = []
    label = []
    bboxs = []
    for i in range(len(images)):
        print("Reading test image: " + images[i][0][0])
        img = cv.imread(img_path + images[i][0][0])
        bbox = get_bboxs(annot_path, annotation[i][0][0])

        for idx in range(bbox.shape[1]):
            imgp = img[bbox[1, idx]: bbox[3, idx], bbox[0, idx]: bbox[2, idx]]
            if imgp.shape[0] == 0 or imgp.shape[1] == 0:
                print("ERROR")
                return 0
            imgp = transform.resize(imgp, (w, h), 1, 'constant')
            imgs.append(imgp)
            label.append(labels[i])

    return np.asarray(imgs, np.float32), np.asarray(label, np.int32), len(np.unique(label))
train = sio.loadmat('../data/train_data.mat')
test = sio.loadmat('../data/test_data.mat')
cell_train = train['train_info'][0][0]
cell_test = test['test_info'][0][0]
data_train, label_train, num_class = read_train('../Annotation/', '../Images/', cell_train)
data_test, label_test, num_class = read_test('../Annotation/', '../Images/', cell_test)

y_train = np.zeros((np.shape(data_train)[0], num_class))
for i in range(np.shape(data_train)[0]):
    y_train[i,label_train[i] - 1] = 1

y_test = np.zeros((np.shape(data_test)[0], num_class))
for i in range(np.shape(data_test)[0]):
    y_test[i,label_test[i] - 1] = 1

# 打乱顺序
num_example = data_train.shape[0]
arr = np.arange(num_example)
np.random.shuffle(arr)
data_train = data_train[arr]
y_train = y_train[arr]


def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    data = []
    label = []
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]



images = tf.placeholder(tf.float32, [None, 224, 224, 3])
true_out = tf.placeholder(tf.float32, [None, num_class])
train_mode = tf.placeholder(tf.bool)
vgg = vgg19.Vgg19_trainable('./vgg19.npy', num_class=num_class)
vgg.build(images, train_mode)
with tf.name_scope('loss'):
    cost = tf.nn.softmax_cross_entropy_with_logits(logits=vgg.fc8, labels=true_out)
    cost = tf.reduce_mean(cost)
    tf.summary.scalar('loss', cost)


global_step = tf.Variable(0, trainable=False)
initial_learning_rate = 0.0001
learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                           global_step=global_step,
                                           decay_steps=100, decay_rate=0.9)
train = tf.train.AdamOptimizer(learning_rate).minimize(cost)
add_global = global_step.assign_add(1)
correct_prediction = tf.nn.in_top_k(vgg.prob, tf.argmax(true_out, 1), 1)
correct_num = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    #     correct_prediction = tf.nn.in_top_k(vgg.prob, tf.argmax(true_out, 1), 10)
    #     correct_num = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
    #     acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    n_epoch = 100
    batch_size = 64
    tf.global_variables_initializer().run()
    # merged
    merged = tf.summary.merge_all()
    VALIDATION_SIZE = 100
    EARLY_STOP_PATIENCE = 10
    best_validation_loss = 100000
    current_epoch = 100
    X_valid, y_valid = data_test[:VALIDATION_SIZE], y_test[:VALIDATION_SIZE]
    # data_train, y_train = data_train[VALIDATION_SIZE:], y_train[VALIDATION_SIZE:]
    writer = tf.summary.FileWriter("logs/", sess.graph)

    print("Number of Sample: %d, Total class: %d" % (data_train.shape[0], num_class))
    for epoch in range(n_epoch):
        start_time = time.time()

        # training
        train_loss, train_acc, n_batch = 0, 0, 0
        # data, label = minibatches(data_train, y_train, batch_size, True)
        for x_train_a, y_train_a in minibatches(data_train, y_train, batch_size, shuffle=True):
            _, err, ac, y, crr = sess.run([train, cost, acc, vgg.prob, correct_prediction],
                                          feed_dict={images: x_train_a, true_out: y_train_a, train_mode: True})
            print(np.where(y_train_a[0] == np.max(y_train[0]))[0][0])
            print(np.where(y[0] == np.max(y[0]))[0][0])
            train_loss = train_loss + err
            train_acc = train_acc + ac
            n_batch = n_batch + 1
            print("it: %d, numbatch: %d, loss: %g,acc: %g" % (epoch, n_batch, err, ac))

        print("   train loss: %f" % (train_loss / n_batch))
        print("   train acc: %f" % (train_acc / n_batch))

        #        vgg_test = vgg19_test.Vgg19('test-save.npy')
        #        vgg_test.build(images)
        #        cost_test = tf.nn.softmax_cross_entropy_with_logits(logits=vgg_test.fc8, labels=true_out)
        #        cost_test = tf.reduce_mean(cost)
        validation_loss, val_acc = sess.run([cost, acc],
                                            feed_dict={images: X_valid, true_out: y_valid, train_mode: False})

        print('epoch %d done! validation loss: %g, accuracy: %g' % (epoch, validation_loss, val_acc))
        if (validation_loss < best_validation_loss) and (validation_loss < 3):
            #            print(validation_loss)
            best_validation_loss = validation_loss
            current_epoch = epoch
            vgg.save_npy(sess, './test-save.npy')  # 即时保存最好的结果
        elif (epoch - current_epoch) >= EARLY_STOP_PATIENCE:
            print('early stopping')
            break

