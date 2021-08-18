#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from glob import glob
import cv2
from tqdm.auto import tqdm
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

train_data_path = "source"
# val_data_path = 'target_s'

batch_size = 32
val_batch_size = 20
epochs = 10
inner_lr = 0.02

n_way = 3
k_shot = 5
q_query = 1

width = 64
hight = 64
input_shape = (width, hight, 1)
outer_lr = 0.0001

# ## Visualize Data
# training v.s test Omniglot classes
source_classes = glob('source/*')
target_classes = glob('target_s/*')
print(f'train: test = {len(source_classes)} : {len(target_classes)}')


# 觀察圖片內容
# path = np.random.choice(glob('source/*/*.JPG'), 1)[0]
# img = cv2.imread(path)
# print(path, img.shape)
# plt.imshow(img)
# plt.show()


def load_img(path, width=width, hight=hight):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255.
    min_of_shape = np.min(image.shape[:2])
    oh = (image.shape[0] - min_of_shape) // 2
    ow = (image.shape[1] - min_of_shape) // 2
    new_size = (width, hight)
    image = image[oh:oh + min_of_shape, ow:ow + min_of_shape]
    return np.expand_dims(cv2.resize(image, new_size), -1)


# ## Prepare the data
class MetaDataLoader:
    def __init__(self, data_path, batch_size, n_way=5, k_shot=1, q_query=1):
        """
        :param data_path: 資料夾下有子資料夾
        :param batch_size: 一個批次有多少個不同的task
        :param n_way: 一個task有幾類
        :param k_shot: 一個類別中有幾張圖片用於inner loop training
        :param q_query: 一個類別中有幾張圖片用於outer loop training
        """
        self.file_list = [f for f in glob(data_path + "**/*", recursive=True)]
        self.steps = len(self.file_list) // batch_size

        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.meta_batch_size = batch_size

    def get_one_task_data(self):
        """
        取一個task, 有n_way類別,每類別有 k_shot於inner training, q_query張於outer training
        :return: support_data, query_data
        """
        img_dirs = random.sample(self.file_list, self.n_way)
        support_data = []
        query_data = []

        support_image = []
        support_label = []
        query_image = []
        query_label = []

        for label, img_dir in enumerate(img_dirs):
            img_list = [f for f in glob(img_dir + "**/*.JPG", recursive=True)]
            images = random.sample(img_list, self.k_shot + self.q_query)
            # Read support set
            for img_path in images[:self.k_shot]:
                support_data.append((load_img(img_path), label))

            # Read query set
            for img_path in images[self.k_shot:]:
                query_image.append(load_img(img_path))
                query_label.append(label)

        # shuffle support set
        random.shuffle(support_data)
        for data in support_data:
            support_image.append(data[0])
            support_label.append(data[1])

        # shuffle query set
        random.shuffle(query_data)
        for data in query_data:
            query_image.append(data[0])
            query_label.append(data[1])

        return np.array(support_image), np.array(support_label), np.array(query_image), np.array(query_label)

    def get_one_batch(self):
        """
        取一個batch, 以task為個體
        :return: k_shot_data, q_query_data
        """
        while True:
            batch_support_image = []
            batch_support_label = []
            batch_query_image = []
            batch_query_label = []

            for _ in range(self.meta_batch_size):
                support_image, support_label, query_image, query_label = self.get_one_task_data()
                batch_support_image.append(support_image)
                batch_support_label.append(support_label)
                batch_query_image.append(query_image)
                batch_query_label.append(query_label)

            yield np.array(batch_support_image), np.array(batch_support_label), \
                  np.array(batch_query_image), np.array(batch_query_label)


train_loader = MetaDataLoader(train_data_path, batch_size, n_way=n_way, k_shot=k_shot, q_query=q_query)
# val_loader = MetaDataLoader(val_data_path, batch_size, n_way=n_way, k_shot=k_shot, q_query=q_query)

# ## Visualize some examples from the dataset
support_imgs, support_labels, q_imgs, q_labels = train_loader.get_one_task_data()


# plt.figure(figsize=(20, 10))
# for i in range(len(support_imgs)):
#     plt.subplot(3, 5, i + 1)
#     plt.title(f'label: {support_labels[i]}')
#     plt.imshow(support_imgs[i, :, :, 0], cmap='gray')
# plt.show()


# ## Build the model
def conv_bn(x):
    x = layers.Conv2D(filters=64, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)
    return x


inputs = layers.Input(shape=input_shape)
x = conv_bn(inputs)
x = conv_bn(x)
x = conv_bn(x)
x = conv_bn(x)
x = layers.Flatten()(x)
outputs = layers.Dense(n_way, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)

inner_optimizer = keras.optimizers.Adam(learning_rate=inner_lr)
outer_optimizer = keras.optimizers.Adam(learning_rate=outer_lr)

model.compile(optimizer=inner_optimizer,
              loss=keras.losses.sparse_categorical_crossentropy)


# ## Train the model
def maml(model, batch_data, inner_optimizer, outer_optimizer, inner_step=1, training=True):
    meta_support_image, meta_support_label, meta_query_image, meta_query_label = next(batch_data)
    batch_acc = []
    batch_loss = []

    for support_image, support_label, query_image, query_label in zip(meta_support_image, meta_support_label,
                                                                      meta_query_image, meta_query_label):
        # 保存一開始model weights
        meta_weights = model.get_weights()

        # inner loop training
        for _ in range(inner_step):
            with tf.GradientTape() as tape:
                logits = model(support_image, training=True)
                loss = keras.losses.sparse_categorical_crossentropy(support_label, logits)
                loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, model.trainable_variables)
            inner_optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # outer loop training: 計算query set loss
        with tf.GradientTape() as tape:
            logits = model(query_image, training=True)
            loss = keras.losses.sparse_categorical_crossentropy(query_label, logits)
            loss = tf.reduce_mean(loss)
            acc = (np.argmax(logits, -1) == query_label).astype(np.int32).mean()

            batch_loss.append(loss)
            batch_acc.append(acc)
        # 回復成一開始weights
        model.set_weights(meta_weights)

        # 在training set中需要更新 meta model
        if training:
            grads = tape.gradient(loss, model.trainable_variables)
            outer_optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return np.array(batch_loss).mean(), np.array(batch_acc).mean()


# #### MAML
train_meta_loss = []
train_meta_acc = []
val_meta_loss = []
val_meta_acc = []

for meta_iter in tqdm(range(epochs)):
    # Training
    for i in tqdm(range(train_loader.steps), leave=False):
        batch_data = train_loader.get_one_batch()  # batch_sup_img, batch_sup_label, batch_query_img, batch_query_label
        batch_train_loss, batch_train_acc = maml(
            model, batch_data, inner_optimizer, outer_optimizer, inner_step=1, training=True)
        train_meta_loss.append(batch_train_loss)
        train_meta_acc.append(batch_train_acc)
    # # Validation
    # for i in range(val_loader.steps):
    #     batch_data = val_loader.get_one_batch()
    #     batch_val_loss, batch_val_acc = maml(model, batch_data, inner_optimizer, outer_optimizer, inner_step=3,
    #                                          training=False)
    #     val_meta_loss.append(batch_val_loss)
    #     val_meta_acc.append(batch_val_acc)
    # print(
    #     f'[epoch {meta_iter:05d}]: train_loss: {np.mean(train_meta_loss):.4f} val_loss: {np.mean(val_meta_loss):.4f} train_acc: {np.mean(train_meta_acc):.3f} val_acc: {np.mean(val_meta_acc):.3f}')
    print(f'[epoch {meta_iter:05d}]: tr_loss: {np.mean(train_meta_loss):.4f}, '
          f'tr_acc: {np.mean(train_meta_acc):.4f}')


# # test
class ts_dataldr:
    def __init__(self, test_dir='./test1.csv'):
        self.df = pd.read_csv(test_dir)

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == len(self.df): raise StopIteration
        row_data = self.df.iloc[self.count]
        support_image = []
        support_label = []
        for index, k in enumerate(['support_0', 'support_1', 'support_2']):
            for img_dir in glob(os.path.join('target_s', row_data[k], '*.JPG'), recursive=True):
                support_image.append(load_img(img_dir))
                support_label.append(index)
        target_data = load_img(os.path.join('target_q', row_data['filename']))
        return np.array(support_image), np.array(support_label), np.array(target_data)[np.newaxis, :]


def ts_maml(ts_dir, model, inner_optimizer, inner_step=1):
    all_result = []
    for support_image, support_label, query_image in ts_dataldr(ts_dir):
        meta_weights = model.get_weights()
        # inner loop training
        for _ in range(inner_step):
            with tf.GradientTape() as tape:
                logits = model(support_image, training=True)
                loss = keras.losses.sparse_categorical_crossentropy(support_label, logits)
                loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, model.trainable_variables)
            inner_optimizer.apply_gradients(zip(grads, model.trainable_variables))
        # query
        all_result.extend(np.argmax(model(query_image, training=True), -1))

        model.set_weights(meta_weights)
    return all_result


final_result = ts_maml('./test1.csv', model, inner_optimizer, inner_step=5)

# Write to csv
df2 = pd.read_csv('SampleSubmission1.csv')
df2['ans'] = final_result
df2.to_csv('Submission1.csv', index=False)
print('Done.')
