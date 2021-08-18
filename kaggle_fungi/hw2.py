#!/usr/bin/env python
# coding: utf-8

# # Relation Network Keras

# 有點像Siamese network中"learn to compare"的概念，Relation Network使用一個embedding network為data做特徵萃取，再使用一個relation network學習依高維空間比較的函式。
# 
# Siamese只負責算出兩個類別的高維特徵關聯性，relation net則將embeded 出的高維資料間的關聯性做softmax，找出最有相關的那一個。這使得Relation net等於能平行處理多個類別比較，加快inference速度。
# 
# 以500人公司做人臉識別的case來講，每個人10張照片，使用Siamese Network要比較至少500次才能完成人員的確認，或者使用500倍記憶體來達成平行化。相較之下Relation network亦可拿來做少way數的N-way-K-shot training，一次比較N組人臉，這樣既能加快比較速度也能減少記憶體消耗。
# 做Relation network的訓練要切分meta-training跟meta-testing classes，像是500個人臉中找出400個做為training、100個做為testing。訓練時每次從400個class中隨機挑選5個class做訓練，每個class 1張照片，其中一個class中多挑選一張作為query data。這樣挑選多次(範例用10000次)後成為一個dataset，以這個dataset做訓練，則可以使model變成一個可以適應各種5-way比較任務的model。在testing時可以從100個testing中隨機挑選五個class做K次訓練並測試，每個class一張照片中要能挑出與query image相同的照片，值得一提的是K可以是0。
# 
# 在forward的時候，Relation net會先將N個class的instances(examples)還有query data做特徵萃取($f_\phi$)，並將一個instance跟query組成一個pair concatenate起來，並進到relation network($g_\phi$)，共取得N個relation score。這個score的值會再被拉到0~1之間(原作者是用sigmoid function)，做1-hot prediction。在backward方面，這個network的output型態與傳統classificaiton一樣，所以也可以使用categorical cross entropy作為loss function。
# 
# Relation network學習了如何在N個data中找到一張同樣class的data，就算是沒train過的class也可以比較。


# ## 範例內容
# 這次的資料集與Siames Network範例中的資料集一樣是使用[AT & T's the Database of Faces](https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html)做人臉辨識範例，40個人每人共有10張不同角度的照片。
# 
# 我們這次的任務是希望network學會只依照片來辨別人臉屬於哪個人：
# 
# 選出5個不同人的照片，並從這五個人的照片中再抽出一張，network必須辨識這一張屬於哪個人。

# **程式碼部分會做以下部分：**
# * Data Loader
# * 資料抽樣、切分
# * 建立模型
# * 模型訓練
# * 預測結果

# # Data Loader
# * (meta)batch: 一次訓練BATCH_SIZE個task
# * task: 40人選5人不重複
# * sample: 5人資料中都是10張選一張當support
# ，還要從剩下45張選一張query

import re
import cv2
import os
from os.path import join
import random
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from functools import partial
from sklearn.model_selection import train_test_split
# '''Tensorflow'''
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras import callbacks as cb
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Lambda, Dense, Dropout, Conv2D, MaxPooling2D, Flatten,Activation,BatchNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.applications.efficientnet import preprocess_input,EfficientNetB0

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth( device=gpu, enable=True)
if len(gpus)>1:
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{len(gpus)-1}"

# 先設定讀取圖片程式，這個資料集圖檔讀取方式需自訂，要輸入圖片的和寬、高、最大值，其中header是圖檔的標頭，讀取時要略過
# 
# 圖片路徑為{orl_faces資料夾}/s1到s40
# 
# 每個subject的圖片放在不同資料夾

# In[2]:


W,H,CH = 64,64,1

'''完整影像讀取程式，含pre-process'''
def load_img(path, width = W):
    img = cv2.imread(path.numpy().decode())[..., 0].astype(np.float32) / 255.
    shape_dst = np.min(img.shape[:2])
    oh = (img.shape[0] - shape_dst) // 2
    ow = (img.shape[1] - shape_dst) // 2
    center_square = np.array([width,width])// 2
    new_size=(width,width)
    
    # cropping + resize
    img = img[oh:oh + shape_dst, ow:ow + shape_dst]
    img=np.expand_dims(cv2.resize(img, new_size),-1)
    return tf.constant(img)
SUFFIX='.JPG'


# 這次要做5 way 3 shot的data loader，且support中每個class都抽一個query
# 
# * support: BATCH_SIZE x (5 x 3) x 長 x 寬 x 顏色頻道(1)
# * query: BATCH_SIZE x 5 x 長 x 寬 x 顏色頻道(1)
# 
# 會先load in set中的的各種class排列組合 shuffle一次
# 
# 再來每個meta batch依照batch size抽取數個排列組合
# 
# 接著依照順序5人中每人抽3張丟進support的位置中，再每人抽一張丟進query的位置中
# support set的label為被抽到的順序，依序是0,1,2,3,4
# query set的label就照著support去配置，看抽到的是前面support中的第幾個人就是幾號，採one hot encoding


WAYS = 3
SHOTS=5
QUERIES=1
BATCH_SIZE=8

np.random.seed(2021)
tf.random.set_seed(2021)
# DATA_ROOT=".."
# DATA_ROOT=join("..","..","data")
# DATA_ROOT="I:\AIA Small Data\datasets"
# all_classes = glob(join("orl_faces","*"))
# source_classes,target_classes=train_test_split(all_classes,test_size=0.2)
# sorce_len=len(source_classes)
# target_len=len(target_classes)
# print(f"total {len(all_classes)} classes=source {sorce_len} + target {target_len} classes")

source_classes = glob('source/*')
sorce_len=len(source_classes)
print(f"source: {sorce_len}")

# 篩掉圖片數不滿足的人
# min([len(glob(join(sdir,'*'+SUFFIX))) for sdir in source_classes])

## exclude classes with too few examples
# source_classes=[sdir for sdir in source_classes if len(glob(join(sdir,'*'+SUFFIX)))>SHOTS+QUERIES]

# sorce_len=len(source_classes)
# print(f"source {sorce_len} + target {target_len} classes")

# min([len(glob(join(sdir,'*'+SUFFIX))) for sdir in source_classes])


# 建立每個人的loader，隨意從每個人的圖片中抽取 (尚未指定張數)
# '''Mapping function for loading'''
map_fun=lambda string: tf.py_function(func=load_img,inp=[string], Tout=tf.float32)
# '''Source set中每個人都有一個tf Dataset loader'''
source_sup_sub = [
    tf.data.Dataset.list_files(glob(join(sc,'*'+SUFFIX)), shuffle=True)
    .map(map_fun).cache()
    for sc in source_classes
]
# source_q_sub = source_sup_sub.copy()
# source_q_sub = [
#     tf.data.Dataset.list_files(glob(join(sc,'*'+SUFFIX)), shuffle=True)
#     .map(map_fun).cache()
#     for sc in source_classes
# ]
# '''Target set中每個人都有一個tf Dataset loader'''
# target_sup_sub = [
#     tf.data.Dataset.list_files(glob(join(sc,'*'+SUFFIX)), shuffle=True)
#     .map(map_fun).cache()
#     for sc in target_classes
# ]
# target_q_sub = [
#     tf.data.Dataset.list_files(glob(join(sc,'*'+SUFFIX)), shuffle=True)
#     .map(map_fun).cache()
#     for sc in target_classes
# ]


# **將所有可能run過一遍，讓cache記得**
for sub in source_sup_sub :
    for x in iter(sub.batch(10)):
        pass
# for sub in source_q_sub :
#     for x in iter(sub.batch(10)):
#         pass

# for sub in target_sup_sub :
#     for x in iter(sub.batch(10)):
#         pass
# for sub in target_q_sub :
#     for x in iter(sub.batch(10)):
#         pass


# 建立每個task的loader，隨意從WAY個人抽取SHOT張，並再隨機指派每個way的class是從0~WAY-1的哪一個
def gen(all_sub):
    order = np.random.permutation(len(all_sub))  # why n shuffle??
    for tasks in range(len(all_sub) // WAYS):
        picked = [all_sub[tt] for tt in order[WAYS * tasks:WAYS * (tasks + 1)]]
        # Support
        support = tf.concat(
            [
                next(
                    iter(
                        sub.batch(SHOTS).prefetch(WAYS)
                    )
                ) for sub in picked
            ]
            , axis=0)
        # 這邊每個task的label都是自己取的，編號從0開始加到WAYS-1
        support_label = tf.repeat(tf.range(WAYS, dtype=tf.float32), SHOTS)
        # Shuffle support
        order2 = np.random.permutation(WAYS * SHOTS)
        support = tf.stack([support[ii] for ii in order2])
        support_label = tf.stack([support_label[ii] for ii in order2])

        # Query
        query_label = np.random.choice(range(WAYS), size=QUERIES, replace=False)
        query = tf.concat(
            [
                next(
                    iter(
                        picked[idx].batch(1).prefetch(1)
                    )
                ) for idx in query_label
            ]
            , axis=0)

        # oh_support_label = keras.utils.to_categorical(support_label, num_classes=WAYS)
        # oh_query_label = keras.utils.to_categorical(query_label, num_classes=WAYS)
        # return tf.concat([support, query], axis=0), \
        #        tf.stack(np.concatenate((oh_support_label,oh_query_label), axis=0))
        # return tf.concat([support, query], axis=0),\
        #        tf.stack([keras.utils.to_categorical(idx, num_classes=WAYS) for idx in query_label], axis=0)
        yield tf.concat([support, query], axis=0), \
               tf.stack([keras.utils.to_categorical(idx, num_classes=WAYS) for idx in query_label], axis=0)


df = pd.read_csv('./test1.csv')
def ts_gen():
    # # 歷遍test 資料 (2200)
    for i in range(len(df)):
        # 每一筆test 有 三個類別
        print(f'test: {i}')
        row_data = df.iloc[i]
        source_sub = []
        for k in ['support_0', 'support_1', 'support_2']:
            md = tf.data.Dataset.list_files(
                file_pattern=os.path.join('target_s', row_data[k], '*.JPG'),
                shuffle=True).map(map_fun)
            source_sub.append(md)

        ts_target_sub = tf.data.Dataset.list_files(
            os.path.join('target_q', row_data['filename']),
            shuffle=True).map(map_fun)

        # # Support
        data = []
        for sub in source_sub:
            data.append(next(iter(sub.batch(SHOTS).prefetch(WAYS))))
        support = tf.concat(data, axis=0)

        support_label = tf.repeat(tf.range(WAYS, dtype=tf.float32), SHOTS)
        # Shuffle support
        order2 = np.random.permutation(WAYS * SHOTS)
        support = tf.stack([support[ii] for ii in order2])
        support_label = tf.stack([support_label[ii] for ii in order2])

        # Query
        query = next(iter(ts_target_sub.batch(1).prefetch(1)))
        # yield (support, query), (support_label,)

        return  tf.concat([support, query], axis=0)

# 建立data generator，可以一次抽一個meta batch的資料
# 1. 要input一個function(這邊用partial產生一個預先設定好參數的function)
# 1. output types要對好前面的格式
# 2. output shape要給對
# g=gen(source_sup_sub)
# g = ts_gen()
MLUT=16
data_source = tf.data.Dataset.from_generator(
    partial(gen,source_sup_sub),
    output_types=(tf.float32,tf.float32),
    output_shapes=((WAYS*SHOTS+QUERIES,W,H,CH),(QUERIES,WAYS))
).repeat(MLUT).shuffle(buffer_size=999).cache().batch(BATCH_SIZE).prefetch(MLUT)
# data_target = tf.data.Dataset.from_generator(
#     ts_gen,
#     output_types=(tf.float32,tf.float32),
#     output_shapes=((WAYS*SHOTS+QUERIES,W,H,CH),)
# ).repeat(MLUT*4).batch(BATCH_SIZE)


# 可以看一下每次sample出來的東西
# * X:共BATCH_SIZE個meta batch,有WAYS*SHOTS+QUERIES個圖，每個圖WxHxCH大小
# * y:共QUERIES個答案


# for source_x,source_y in data_source:
#     break


# plt.figure(figsize=(20,6));ee=1
# batch=0
# print(f"Support 0")
# for jj in range(WAYS*SHOTS):
#     plt.subplot(SHOTS,WAYS*2,ee);plt.imshow(source_x[batch,jj,...,0],"gray")
#     plt.title(f"{jj//SHOTS}");plt.axis("off");ee+=1
# plt.show()
# print("Query 0");plt.figure(figsize=(10,2));ee=1
# for jj in range(QUERIES):
#     plt.subplot(1,QUERIES,ee);plt.imshow(source_x[batch,WAYS*SHOTS+jj,...,0],"gray")
#     plt.title(f"{source_y[batch,jj].numpy().argmax(-1)}");plt.axis("off");ee+=1
# plt.show()


for source_x,source_y in data_source:
    break
print(source_x.shape,source_x.numpy().min(),source_x.numpy().max())
print(source_y.shape,source_y.numpy().min(),source_y.numpy().max())
print(source_y.numpy().argmax(-1))

# for target_x,target_y in data_target:
#     break
# print(target_x.shape,target_x.numpy().min(),target_x.numpy().max())
# print(target_y.shape,target_y.numpy().min(),target_y.numpy().max())
# print(target_y.numpy().argmax(-1))


# # 建立模型
# 按圖索驥，開始組裝network，會有base netwok(for embedding)還有relation network。
# 
# * base netwok：由convolution networks組成。
# 
# * relation net：用convolution network以及densenet組成，output使用sigmoid讓數值介於0~1之間。

# Embedding function
def conv_net(input_shape):
    convnet = Sequential()
    for i in range(3):
        convnet.add(Conv2D(64,(3,3),padding='valid',input_shape=input_shape))
        convnet.add(BatchNormalization())
        convnet.add(Activation('relu'))
        convnet.add(MaxPooling2D())
    return convnet
def build_relation_network(input_shape):
    seq = Sequential()
    #layer1
    seq.add(Conv2D(64, kernel_size=3, input_shape=input_shape,
                           padding="valid",activation='relu'))
    seq.add(BatchNormalization()) 
    seq.add(MaxPooling2D(pool_size=(2, 2)))  
#     seq.add(Dropout(.1))
    
    seq.add(Flatten())
    seq.add(Dense(8, activation='relu'))
#     seq.add(Dropout(0.1))
    seq.add(Dense(1, activation=None))
    return seq


# 接著組裝network。
# 
# 這個例子裏面給query用的input再加上每個class的example共有6個，每組訓練資料都要讀取6張圖。
# 
# 接著將每張圖拿去base_network做成feature maps，
# 
# 這個network的output與傳統classificaiton一樣，所以也可以使用categorical cross entropy作為loss function。

base_dim = (W,H,CH)
base_network = conv_net(base_dim)
# Query feature
x_in=Input(shape=(WAYS*SHOTS+QUERIES,W,H,CH))
latent_s=[base_network(x_in[:,ii]) for ii in range(WAYS*SHOTS)]
latent_q=[base_network(x_in[:,WAYS*SHOTS+ii]) for ii in range(QUERIES)]

relation_net=build_relation_network((latent_q[0].shape[-3],latent_q[0].shape[-2],latent_q[0].shape[-1]*2))

y=[]
for q in latent_q:
    relation_score=[]
    for ww in range(WAYS):
        relation=[relation_net(tf.concat([q,s],-1)) for s in latent_s[ww*SHOTS:(ww+1)*SHOTS]]
        relation_score.append(tf.reduce_mean(tf.concat(relation,-1),-1,keepdims=True))
        
    y.append(tf.nn.softmax(tf.concat(relation_score,-1),-1))
pred=tf.stack(y,1)

model = Model(inputs=x_in, outputs=pred)


lr=0.001
reduce_lr = cb.ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=6, min_lr=1e-8, verbose=1)
tensorboard = cb.TensorBoard(log_dir="tf_relation_logs")
earlystop=cb.EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True, verbose=1)
opt = tf.keras.optimizers.Adam(lr=lr)
model.compile(loss=CategoricalCrossentropy(), optimizer=opt, metrics=['acc'])


# # 模型訓練
# 訓練使用10 epoch、batch size為32個tasks，
# 在訓練時再使用validation split驗證loss是否下降，或者只是over fitting。


EPOCHS=10 #400
# %%time
try:
    model.fit(data_source,
              epochs=EPOCHS, verbose=1,workers=4,
              callbacks=[reduce_lr,earlystop, tensorboard],
              validation_data=data_source)
except KeyboardInterrupt:
    print("KeyboardInterrupt")


# # 預測結果
# model.input_shape
# model.output_shape

eva = model.evaluate(data_target, verbose=2)
pre = model.predict(data_target)

for i ,j in data_target:
    # print(i.shape)
    print(j.shape)
