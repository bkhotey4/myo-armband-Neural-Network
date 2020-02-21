# This is Myo Armband via neural network model 
# CopyRight @ wmlab & NCU 2019 Apr

import collections
import myo 
import threading
import time
from time import sleep
import numpy as np
import tensorflow as tf
from include.model import model
from myo import Hub
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'    #这是一个警告，并不影响tensorflow运行，上面这段代码表示忽略这个警告，和tensorflow的硬件加速有关。

x, y, output, global_step, y_pred_cls = model(5)

saver = tf.train.Saver()
_SAVE_PATH = "./data/tensorflow_sessions/myo_armband/auther_20190513model/"
sess = tf.Session()

status = 0
X = []

try:
    print("Trying to restore last checkpoint ...")
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=_SAVE_PATH)
    print(last_chk_path)
    saver.restore(sess, save_path=last_chk_path)
    print("Restored checkpoint from:", last_chk_path)
except Exception as e:
    print("Failed to restore checkpoint. Initializing variables instead.")
    sess.run(tf.global_variables_initializer())

class MyListener(myo.DeviceListener):

    def __init__(self, queue_size=8):
        self.lock = threading.Lock()
        self.emg_data_queue = collections.deque(maxlen=queue_size)
        self.ori_data_queue = collections.deque(maxlen=queue_size)

    def on_connected(self, event):        
        event.device.stream_emg(True)
        
    def on_emg(self, event):
        #with self.lock:
        if(status):
            X.append(event.emg)
            self.emg_data_queue.append((event.timestamp,event.emg))

    def get_emg_data(self):
        with self.lock:
            return list(self.emg_data_queue)

myo.init()
hub = myo.Hub()
feed = myo.ApiDeviceListener()
start = time.time()
temp = []

status = 9999

listener = MyListener()
with hub.run_in_background(listener.on_event):
    req_iter = 20
    #regis_x = 999 #暫存檢查重複值
    while(1):
        if len(X) >= 64:
            #if regis_x != X[0][0]: 
                #print(len(X))
            X_temp = X[0:8]
                #print('qwe')
                #print(X[0:2])
                #print('qwe')
                #print(len(X_temp))
            X_temp = list(np.stack(X_temp).flatten())
                #print(X_temp[0:2]) 
                #regis_x = X[0][0]
            pred = sess.run(y_pred_cls, feed_dict={x: np.array([X_temp])})
            temp.append(pred[0])
            X = []
            
        if time.time() - start >= 1:
            #pred = sess.run(y_pred_cls, feed_dict={x: np.array([X_temp])})
            #temp.append(pred[0])
            response = np.argmax(np.bincount(temp))
            print("Predicted gesture: {0}".format(response))
            temp = []
            start = time.time()
    sleep(1)

 
# try:
#     listener = MyListener()
#     hub.run(listener,2000)
#     #hub.run_in_background(listener.on_event)
#     while True:
#         data = listener.get_emg_data()

     
#         if len(data) > 0:
#             tmp = []
#             for v in listener.get_emg_data():
#                 tmp.append(v[1])
#             print('oneone')
#             print(tmp)
#             tmp = list(np.stack(tmp).flatten())
#             print(tmp)
#             print(y_pred_cls)
#             if len(tmp) >= 64:
#                 pred = sess.run(y_pred_cls, feed_dict={x: np.array([tmp])})
#                 temp.append(pred[0])
#                 #print(temp)
#         if time.time() - start >= 1:
#             response = np.argmax(np.bincount(temp))
#             #print(temp)
#             print("Predicted gesture: {0}".format(response))
#             temp = []
#             start = time.time()
#         time.sleep(0.01)
# finally:
#     #hub.shutdown()
#     sess.close()

#參考資料:https://github.com/NiklasRosenstein/myo-python/issues/68
