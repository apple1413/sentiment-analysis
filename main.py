# encoding:utf-8

import numpy as np
import tensorflow as tf
# a = tf.Variable([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], [[11, 12, 13, 14, 15],[6, 7, 8, 9, 10]]])
# index_a = tf.Variable([0, 2])
# tf.global_variables_initializer()
# c=tf.gather(a, 2)
#
#
#
flags=tf.flags

flags.DEFINE_bool("do_pre", True, "Whether to run prected."
)
FLAGS = flags.FLAGS
wordsList = np.load('wordsList.npy')
print('载入word列表')
wordsList = wordsList.tolist()
wordsList = [word.decode('UTF-8')
             for word in wordsList]
wordVectors = np.load('wordVectors.npy')
print('载入文本向量')

print(len(wordsList))
print(wordVectors.shape)

import os
from os.path import isfile, join

pos_files = ['pos/' + f for f in os.listdir(
    'pos/') if isfile(join('pos/', f))]
neg_files = ['neg/' + f for f in os.listdir(
    'neg/') if isfile(join('neg/', f))]
num_words = []
for pf in pos_files:
    with open(pf, "r", encoding='utf-8') as f:
        line = f.readline()
        counter = len(line.split())
        num_words.append(counter)
print('正面评价完结')

for nf in neg_files:
    with open(nf, "r", encoding='utf-8') as f:
        line = f.readline()
        counter = len(line.split())
        num_words.append(counter)
print('负面评价完结')

num_files = len(num_words)
print('文件总数', num_files)
print('所有的词的数量', sum(num_words))
print('平均文件词的长度', sum(num_words) / len(num_words))

import re

strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
num_dimensions = 300  # Dimensions for each word vector


def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())


max_seq_num = 250



"""
ids = np.zeros((num_files, max_seq_num), dtype='int32')
file_count = 0
for pf in pos_files:
  with open(pf, "r", encoding='utf-8') as f:
    indexCounter = 0
    line = f.readline()
    cleanedLine = cleanSentences(line)
    split = cleanedLine.split()
    for word in split:
      try:
        ids[file_count][indexCounter] = wordsList.index(word)
      except ValueError:
        ids[file_count][indexCounter] = 399999  # 未知的词
      indexCounter = indexCounter + 1
      if indexCounter >= max_seq_num:
        break
    file_count = file_count + 1

for nf in neg_files:
  with open(nf, "r",encoding='utf-8') as f:
    indexCounter = 0
    line = f.readline()
    cleanedLine = cleanSentences(line)
    split = cleanedLine.split()
    for word in split:
      try:
        ids[file_count][indexCounter] = wordsList.index(word)
      except ValueError:
        ids[file_count][indexCounter] = 399999  # 未知的词语
      indexCounter = indexCounter + 1
      if indexCounter >= max_seq_num:
        break
    file_count = file_count + 1

np.save('idsMatrix', ids)
"""

from random import randint

batch_size = 24
lstm_units = 64
num_labels = 2
iterations = 100
lr = 0.001
ids = np.load('idsMatrix.npy')
# line = f.readline()
# cleanedLine = cleanSentences(line)
# split = cleanedLine.split()
# indexCounter = 0
# for word in split:
#     try:
#         ids[25000][indexCounter] = wordsList.index(word)
#     except ValueError:
#         ids[25000][indexCounter] = 399999  # 未知的词
#     indexCounter = indexCounter + 1
#     if indexCounter >= max_seq_num:
#         break


def get_train_batch():
    labels = []
    arr = np.zeros([batch_size, max_seq_num])
    for i in range(batch_size):
        if (i % 2 == 0):
            num = randint(1, 11499)
            labels.append([1, 0])
        else:
            num = randint(13499, 24999)
            labels.append([0, 1])
        arr[i] = ids[num - 1:num]
    return arr, labels


def get_test_batch():
    labels = []
    arr = np.zeros([batch_size, max_seq_num])
    for i in range(batch_size):
        num = randint(11499, 13499)
        if (num <= 12499):
            labels.append([1, 0])
        else:
            labels.append([0, 1])
        arr[i] = ids[num]
    return arr, labels
def get_pred_batch():
    labels = []
    arr = np.zeros([batch_size, max_seq_num])
    for i in range(batch_size):
        num = randint(11499, 13499)
        if (num <= 12499):
            labels.append([1, 0])
        else:
            labels.append([0, 1])
        arr[i] = ids[num]
    return arr, labels

import tensorflow as tf

tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batch_size, num_labels])
input_data = tf.placeholder(tf.int32, [batch_size, max_seq_num])
data = tf.Variable(
    tf.zeros([batch_size, max_seq_num, num_dimensions]), dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors, input_data)

lstmCell = tf.contrib.rnn.BasicLSTMCell(lstm_units)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.5)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

weight = tf.Variable(tf.truncated_normal([lstm_units, num_labels]))#正态分布随机数
bias = tf.Variable(tf.constant(0.1, shape=[num_labels]))
value = tf.transpose(value, [1, 0, 2])#最外层两维进行转置 –代表的是最外层的一维, 1–代表外向内数第二维, 2–代表最内层的一维
last = tf.gather(value, int(value.get_shape()[0])-1) #用一个一维的索引数组，将张量中对应索引的向量提取出来
prediction = (tf.matmul(last, weight) + bias)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

saver = tf.train.Saver()

with tf.Session() as sess:
    if os.path.exists("models") and os.path.exists("models/checkpoint"):
        saver.restore(sess, tf.train.latest_checkpoint('models'))
    else:
        if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
            init = tf.initialize_all_variables()
        else:
            init = tf.global_variables_initializer()
        sess.run(init)
    iterations = 100
    for step in range(iterations):
        next_batch, next_batch_labels = get_test_batch()
        next_batch_train, next_batch_labels_train= get_train_batch()
        print("step:", step, " loss:", (sess.run(
            loss, feed_dict={input_data: next_batch_train, labels: next_batch_labels})))

        if step % 20 == 0:
            print("step:", step, " 测试集正确率:", (sess.run(
                  accuracy, feed_dict={input_data: next_batch})) )
            c=tf.argmax(prediction, 1)


    if not os.path.exists("models"):
        os.mkdir("models")
    save_path = saver.save(sess, "models/model.ckpt")
    print("Model saved in path: %s" % save_path)
