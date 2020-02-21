# This is Myo Armband via neural network model 
# CopyRight @ wmlab & NCU 2019 Apr

import numpy as np
import tensorflow as tf
import csv 
from time import time
from include.data import get_data_set
from include.model import model
import os
from math import sqrt #Python3 program to calculate Root Mean Square
from sklearn.metrics import mean_squared_error

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'   #这是一个警告，并不影响tensorflow运行，上面这段代码表示忽略这个警告，和tensorflow的硬件加速有关。

train_x, train_y = get_data_set()

_BATCH_SIZE = 20    #原本300
_CLASS_SIZE = 5       #有5個手勢
_SAVE_PATH = "./data/tensorflow_sessions/myo_armband/Kaohsiung Chang Gung Memorial Hospital Model/"


x, y, output, global_step, y_pred_cls = model(_CLASS_SIZE)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y)) 
        #tf.reduce_mean 函数用于计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值，主要用作降维或者计算tensor（图像）的平均值。
        #第一個參數logits:就是神經網路最後一層的輸出如果有batch的话，它的大小就是[batchsize，num_classes]，单样本的话，大小就是num_classes
        #第二個labels：实际的标签，大小同上
tf.summary.scalar("Loss", loss)    #一般在画loss,accuary时会用到这个函数又是或說定義 'loss' 與下面要使用的 Optimizer
#optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(loss, global_step=global_step)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step=global_step)
        #創建RMSProp算法优化器，RMSProp被证明有效且实用的深度学习网络优化算法。

correct_prediction = tf.equal(y_pred_cls, tf.argmax(y, dimension=1)) #tf.argmax(y_,1) 代表正确的标签，我们可以用 tf.equal 来检测我们的预测是否真实标签匹配(索引位置一样表示匹配)。
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #第29行和第30行為在實際執行程式之前，我們還需要一些程式碼來評估，最後訓練出來的模型，拿來預測 test data 的準確率如何
        #tf.cast此函数是类型转换函数
tf.summary.scalar("Accuracy/train", accuracy)

init = tf.global_variables_initializer() #将所有全局变量的初始化器汇总，并对其进行初始化
merged = tf.summary.merge_all() #將視覺化輸出，換句話說就是將所有要顯示再tensorboard的資料整合
saver = tf.train.Saver()
sess = tf.Session()  # 初始化 Graph
train_writer = tf.summary.FileWriter(_SAVE_PATH, sess.graph) #將整合好的資料寫入log檔。 位置為當前執行目錄底下的_SAVE_PATH資料夾


try:
    print("Trying to restore last checkpoint ...")
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=_SAVE_PATH)
    saver.restore(sess, save_path=last_chk_path)
    print("Restored checkpoint from:", last_chk_path)
except:
    print("Failed to restore checkpoint. Initializing variables instead.")
    sess.run(tf.global_variables_initializer())


def train(num_iterations = 1000):
    #迭代训练
    for i in range(num_iterations):
        randidx = np.random.randint(len(train_x), size=_BATCH_SIZE)
        batch_xs = train_x[randidx]
        batch_ys = train_y[randidx]

        start_time = time()
        i_global, _ = sess.run([global_step, optimizer], feed_dict={x: batch_xs, y: batch_ys})
        duration = time() - start_time

        if (i_global % 10 == 0) or (i == num_iterations - 1):
            _loss, batch_acc = sess.run([loss, accuracy], feed_dict={x: batch_xs, y: batch_ys})
            msg = "Global Step: {0:>6}, accuracy: {1:>6.1%}, loss = {2:.2f} ({3:.1f} examples/sec, {4:.2f} sec/batch)"
            print(msg.format(i_global, batch_acc, _loss, _BATCH_SIZE / duration, duration))                     

        if (i_global % 100 == 0) or (i == num_iterations - 1):
            data_merged, global_1 = sess.run([merged, global_step], feed_dict={x: batch_xs, y: batch_ys})
            train_writer.add_summary(data_merged, global_1)
            saver.save(sess, save_path=_SAVE_PATH, global_step=global_step)
            print("Saved checkpoint.")
    print()

train(75000)    #30000

sess.close()

#筆記:epochs被定义为向前和向后传播中所有批次的单次训练迭代。这意味着1个周期是整个输入数据的单次向前和向后传递。简单说，epochs指的就是训练过程中数据将被“轮”多少次，就这样。
# one epoch = numbers of iterations(iterations意思是迭代) = N = 训练样本的数量/batch_size。
