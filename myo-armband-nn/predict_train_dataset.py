# This is Myo Armband via neural network model 
# CopyRight @ wmlab & NCU 2019 Apr

import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from include.model import model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'    #这是一个警告，并不影响tensorflow运行，上面这段代码表示忽略这个警告，和tensorflow的硬件加速有关。
import time
# from sklearn.metrics import mean_squared_error
from math import sqrt
#from include.data import get_data_set
from include.testdata import get_data_set
import csv
x, y, output, global_step, y_pred_cls = model(5) #model的數字5代表5個動作

test_x, test_y = get_data_set()
# print('testtesttest')
#print(test_x)
#print(test_y.shape)
# print(test_x[1].shape)
test_l = ["Spread Finger", "Wave Out", "Wave In", "Fist", "Rest"]


saver = tf.train.Saver()
_SAVE_PATH = "./data/tensorflow_sessions/myo_armband/Kaohsiung Chang Gung Memorial Hospital Model/"
sess = tf.Session()


try:
    #print("Trying to restore last checkpoint ...")
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=_SAVE_PATH)
    print("Trying to restore last checkpoint :",last_chk_path)
    saver.restore(sess, save_path=last_chk_path)
    print("Restored checkpoint from :", last_chk_path)
except:
    print("Failed to restore checkpoint. Initializing variables instead.")
    sess.run(tf.global_variables_initializer())


i = 0
predicted_class = np.zeros(shape=len(test_x), dtype=np.int)
while i < len(test_x):
    j = min(i + 300, len(test_x))
    batch_xs = test_x[i:j, :]
    batch_ys = test_y[i:j, :]

    predicted_class[i:j] = sess.run(y_pred_cls, feed_dict={x: batch_xs, y: batch_ys})

    i = j

correct = (np.argmax(test_y, axis=1) == predicted_class)
acc = correct.mean()*100
correct_numbers = correct.sum()
print("-----------------------------------------")
print("Accuracy on Test-DataSet: {0:.2f}% ({1} / {2})".format(acc, correct_numbers, len(test_x)))
print("-----------------------------------------")

cm = confusion_matrix(y_true=np.argmax(test_y, axis=1), y_pred=predicted_class)
for i in range(5):    #range的數字5代表5個動作
    class_name = "({}) {}".format(i, test_l[i])
    print(cm[i, :], class_name)
class_numbers = [" ({0})".format(i) for i in range(5)]
print("".join(class_numbers))

#print(cm[i-1,:])
#print(cm[0][1])
# print(sum(cm[i-1,:]))
print("-----------------------------------------")
if acc >=50 and acc <100:
    print("Body Condition: Very Health, no problem")
elif acc >=40 and acc <50:
    print("Body Condition: Health,but you must follow up your health")
elif acc >=25 and acc <40:
    print("Body Condition: Probably Sick")
elif acc >0 and acc <25:
    print("Body Condition: Really Sick")
print("-----------------------------------------")
print("Spread Finger correct rate (手掌張開正確率) : {0:.3f}%".format(cm[0][0]/sum(cm[i-4,:])*100))
print("-----------------------------------------")
print("Wave Out correct rate (手掌右擺正確率) : {0:.3f}%".format(cm[1][1]/sum(cm[i-3,:])*100))
print("-----------------------------------------")
print("Wave In correct rate (手掌左擺正確率) : {0:.3f}%".format(cm[2][2]/sum(cm[i-2,:])*100))
print("-----------------------------------------")
print("Fist correct rate (手掌握拳正確率) : {0:.3f}%".format(cm[3][3]/sum(cm[i-1,:])*100))
print("-----------------------------------------")
print("Rest correct rate (手掌放鬆正確率) : {0:.3f}%".format(cm[4][4]/sum(cm[i,:])*100))
print("-----------------------------------------")

sess.close()
