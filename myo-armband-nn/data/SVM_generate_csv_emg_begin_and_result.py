# This is Myo Armband svm model 
# CopyRight @ wmlab & NCU 2019 Mar
from __future__ import print_function 
from time import sleep 
from sklearn import svm         
from sklearn import datasets    
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
#from sklearn.manifold import Isomap
#import matplotlib.pyplot as plt      #載入matplotlib
import sklearn.ensemble
import numpy as np
#import panadas as pd
import threading
import collections
import math
import csv               
import myo
from sklearn.metrics import confusion_matrix

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def rms(array):
    n = len(array)
    sum = 0
    for a in array:
        sum =+ a*a
    return np.sqrt((1/float(n))*sum)

def iav(array):
    sum = 0
    for a in array:
        sum += np.abs(a)
    return sum

def ssi(array):
    sum = 0
    for a in array:
        sum += a*a
    return sum

def var(array):
    n = len(array)
    sum = 0
    for a in array:
        sum += a*a
    return ((1/float(n-1))*sum)

def tm3(array):
    n = len(array)
    print('n : ', n)
    sum = 0
    for a in array:
        sum =+ a*a*a
    return np.power((1/float(n))*sum,1/float(3))

def wl(array):  #window length
    sum = 0
    for a in range(0,len(array)-1):
        sum =+ array[a+1] - array[a]
    return sum

def aac(array):
    n = len(array)
    sum = 0
    for a in range(0,n-1):
        sum =+ array[0+1] - array[0]
    return sum/float(n)

def featurize(array):                               #featurize 是作者自定義的
    n = []
    for a in array:
        n.append(rms(a))                       
    return n

status = 0

X = []

def toEuler(quat):
    quat = quat[0]

    # Roll
    sin = 2.0 * (quat.w * quat.w + quat.y * quat.z)
    cos = +1.0 - 2.0 * (quat.x * quat.x + quat.y * quat.y)
    roll = math.atan2(sin, cos)

    # Pitch
    pitch = math.asin(2 * (quat.w * quat.y - quat.z * quat.x))

    # Yaw
    sin = 2.0 * (quat.w * quat.z + quat.x * quat.y)
    cos = +1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
    yaw = math.atan2(sin, cos)
    return [pitch, roll, yaw]

class Listener(myo.DeviceListener):      
    def __init__(self, queue_size=1):
        self.lock = threading.Lock()
        self.emg_data_queue = collections.deque(maxlen=queue_size)
        self.ori_data_queue = collections.deque(maxlen=queue_size)

    def on_connected(self, event):        
        event.device.stream_emg(True)

    def on_emg(self, event):
        #print(np.asarray(event.emg))
        if(status):
            X.append(np.asarray(event.emg))
            #print(X)

    def on_orientation(self, event):
        #print("Orientation:", event.orientation.x, event.orientation.y, event.orientation.z, event.orientation.w)
        with self.lock:
            self.ori_data_queue.append(event.orientation)

    def get_ori_data(self):
        with self.lock:
            return list(self.ori_data_queue)

myo.init()
hub = myo.Hub()
feed = myo.ApiDeviceListener()
listener = Listener()
with hub.run_in_background(listener.on_event):
    status = 9999
    sleep(1)
    myX = []
    req_iter = 20
    train_1 = []
    train_2 = []
    train_3 = []
    train_4 = []
    train_5 = []
    # train_6 = []
    # train_7 = []
    # train_8 = []
    # train_9 = []
    # train_10 = []
    test1=[]
    #ges1 = ['Rock', 'Paper', 'Scissors', 'Lizard', 'Spock']
    #ges2 = ['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']
    #ges3 = ['Spread Fingers', 'Wave Out', 'Wave In', 'Fist', 'Rest','Agree','Pointer','Supination','pronation','Fine Grip']
    ges3 = ['手掌張開', '手掌往右擺', '手掌往左擺', '握拳', '放鬆']
    ges = ges3
    
    f = open('20190716_sEmg_data.csv', 'a',newline='')
    writer = csv.writer(f)
    writer.writerow(['channel1', 'channel2', 'channel3', 'channel4', 'channel5', 'channel6', 'channel7', 'channel8','label'])

    for a in range(1,4):   #次數

        print("\n請準備好您的手勢 -- ", ges[0]," : 準備好了嗎?")
        input("請按Enter鍵繼續...")
        X = []
        while(1):
            if len(X) > 20:
                # print(X[-1])
                train_1.append(np.asarray(X))
                X = []
                if len(train_1) > a*req_iter:
                    break
        # with open('SpreadFingerdataemg.csv', 'a',newline='') as f:   
        #      #configure writer to write standard csv file
        #     writer = csv.writer(f)
        #     writer.writerow(['channel1', 'channel2', 'channel3', 'channel4', 'channel5', 'channel6', 'channel7', 'channel8','label'])
        #     for item in train_1:
        #         # writer.writerow([ item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7]])
        #         for row in item:
        #             writer.writerow([row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],'1'])
        # 彙整全部Emgdata csv :
        # #with open('SpreadFingerdataemg.csv', 'a',newline='') as f:   
        # f = open('20190422EmgData.csv', 'a',newline='')
        #      #configure writer to write standard csv file
        # writer = csv.writer(f)
        # writer.writerow(['channel1', 'channel2', 'channel3', 'channel4', 'channel5', 'channel6', 'channel7', 'channel8','label'])
        for item in train_1:
                # writer.writerow([ row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],'1'])
            for row in item:
                writer.writerow([row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],'1'])   


        print("\n請準備好您的手勢 -- ", ges[1]," : 準備好了嗎?")
        input("請按Enter鍵繼續...")
        X = []
        while(1):
            if len(X) > 20:
                # print(X[-1])
                train_2.append(np.asarray(X))
                X = []
                if len(train_2) > a*req_iter:
                    break
        # with open('Waveoutdataemg.csv', 'a',newline='') as f:   
        #      #configure writer to write standard csv file
        #     writer = csv.writer(f)
        #     writer.writerow(['channel1', 'channel2', 'channel3', 'channel4', 'channel5', 'channel6', 'channel7', 'channel8','label'])
        #     for item in train_2:
        #         # writer.writerow([ item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7]])
        #         for row in item:
        #             writer.writerow([row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],'2'])
        for item in train_2:
                #writer.writerow([ item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7]])
            for row in item:
                writer.writerow([row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],'2'])


        print("\n請準備好您的手勢 -- ", ges[2]," : 準備好了嗎?")
        input("請按Enter鍵繼續...")
        X = []
        while(1):
            if len(X) > 20:
                # print(X[-1])
                train_3.append(np.asarray(X))
                X = []
                if len(train_3) > a*req_iter:
                    break
        # with open('WaveIndataemg.csv', 'a',newline='') as f:   
        #      #configure writer to write standard csv file
        #     writer = csv.writer(f)
        #     writer.writerow(['channel1', 'channel2', 'channel3', 'channel4', 'channel5', 'channel6', 'channel7', 'channel8','label'])
        #     for item in train_3:
        #         # writer.writerow([ item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7]])
        #         for row in item:
        #             writer.writerow([row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],'3'])
        for item in train_3:
                #writer.writerow([ item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7]])
            for row in item:
                    writer.writerow([row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],'3'])

        print("\n請準備好您的手勢 -- ", ges[3]," : 準備好了嗎?")
        input("請按Enter鍵繼續...")
        X = []
        while(1):
            if len(X) > 20:
                # print(X[-1])
                train_4.append(np.asarray(X))
                X = []
                if len(train_4) > a*req_iter:
                    break
        # with open('Fistdataemg.csv', 'a',newline='') as f:   
        #      #configure writer to write standard csv file
        #     writer = csv.writer(f)
        #     writer.writerow(['channel1', 'channel2', 'channel3', 'channel4', 'channel5', 'channel6', 'channel7', 'channel8','label'])
        #     for item in train_4:
        #         # writer.writerow([ item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7]])
        #         for row in item:
        #             writer.writerow([row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],'4'])
        for item in train_4:
                #writer.writerow([ item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7]])
            for row in item:
                    writer.writerow([row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],'4'])


        print("\n請準備好您的手勢 -- ", ges[4]," : 準備好了嗎?")
        input("請按Enter鍵繼續...")
        X = []
        while(1):
            if len(X) > 20:
                # print(X[-1])
                train_5.append(np.asarray(X))
                X = []
                if len(train_5) > a*req_iter:
                    break
        # with open('Restdataemg.csv', 'a',newline='') as f:   
        #      #configure writer to write standard csv file
        #     writer = csv.writer(f)
        #     writer.writerow(['channel1', 'channel2', 'channel3', 'channel4', 'channel5', 'channel6', 'channel7', 'channel8','label'])
        #     for item in train_5:
        #         # writer.writerow([ item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7]])
        #         for row in item:
        #             writer.writerow([row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],'5'])
        for item in train_5:
                #writer.writerow([ item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7]])
            for row in item:
                    writer.writerow([row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],'5'])

        # print("\nGesture -- ", ges[5]," : Ready?")
        # input("Press Enter to continue...")
        # X = []
        # while(1):
        #     if len(X) > 20:
        #         # print(X[-1])
        #         train_6.append(np.asarray(X))
        #         X = []
        #         if len(train_6) > a*req_iter:
        #             break
        # # with open('Waveoutdataemg.csv', 'a',newline='') as f:   
        # #      #configure writer to write standard csv file
        # #     writer = csv.writer(f)
        # #     writer.writerow(['channel1', 'channel2', 'channel3', 'channel4', 'channel5', 'channel6', 'channel7', 'channel8','label'])
        # #     for item in train_2:
        # #         # writer.writerow([ item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7]])
        # #         for row in item:
        # #             writer.writerow([row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],'2'])
        # for item in train_6:
        #         #writer.writerow([ item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7]])
        #     for row in item:
        #         writer.writerow([row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],'6'])

        # print("\nGesture -- ", ges[6]," : Ready?")
        # input("Press Enter to continue...")
        # X = []
        # while(1):
        #     if len(X) > 20:
        #         # print(X[-1])
        #         train_7.append(np.asarray(X))
        #         X = []
        #         if len(train_7) > a*req_iter:
        #             break
        # # with open('Waveoutdataemg.csv', 'a',newline='') as f:   
        # #      #configure writer to write standard csv file
        # #     writer = csv.writer(f)
        # #     writer.writerow(['channel1', 'channel2', 'channel3', 'channel4', 'channel5', 'channel6', 'channel7', 'channel8','label'])
        # #     for item in train_2:
        # #         # writer.writerow([ item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7]])
        # #         for row in item:
        # #             writer.writerow([row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],'2'])
        # for item in train_7:
        #         #writer.writerow([ item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7]])
        #     for row in item:
        #         writer.writerow([row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],'7'])
        
        # print("\nGesture -- ", ges[7]," : Ready?")
        # input("Press Enter to continue...")
        # X = []
        # while(1):
        #     if len(X) > 20:
        #         # print(X[-1])
        #         train_8.append(np.asarray(X))
        #         X = []
        #         if len(train_8) > a*req_iter:
        #             break
        # # with open('Waveoutdataemg.csv', 'a',newline='') as f:   
        # #      #configure writer to write standard csv file
        # #     writer = csv.writer(f)
        # #     writer.writerow(['channel1', 'channel2', 'channel3', 'channel4', 'channel5', 'channel6', 'channel7', 'channel8','label'])
        # #     for item in train_2:
        # #         # writer.writerow([ item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7]])
        # #         for row in item:
        # #             writer.writerow([row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],'2'])
        # for item in train_8:
        #         #writer.writerow([ item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7]])
        #     for row in item:
        #         writer.writerow([row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],'8'])
        
        # print("\nGesture -- ", ges[8]," : Ready?")
        # input("Press Enter to continue...")
        # X = []
        # while(1):
        #     if len(X) > 20:
        #         # print(X[-1])
        #         train_9.append(np.asarray(X))
        #         X = []
        #         if len(train_9) > a*req_iter:
        #             break
        # # with open('Waveoutdataemg.csv', 'a',newline='') as f:   
        # #      #configure writer to write standard csv file
        # #     writer = csv.writer(f)
        # #     writer.writerow(['channel1', 'channel2', 'channel3', 'channel4', 'channel5', 'channel6', 'channel7', 'channel8','label'])
        # #     for item in train_2:
        # #         # writer.writerow([ item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7]])
        # #         for row in item:
        # #             writer.writerow([row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],'2'])
        # for item in train_9:
        #         #writer.writerow([ item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7]])
        #     for row in item:
        #         writer.writerow([row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],'9'])
        
        # print("\nGesture -- ", ges[9]," : Ready?")
        # input("Press Enter to continue...")
        # X = []
        # while(1):
        #     if len(X) > 20:
        #         # print(X[-1])
        #         train_10.append(np.asarray(X))
        #         X = []
        #         if len(train_10) > a*req_iter:
        #             break
        # # with open('Waveoutdataemg.csv', 'a',newline='') as f:   
        # #      #configure writer to write standard csv file
        # #     writer = csv.writer(f)
        # #     writer.writerow(['channel1', 'channel2', 'channel3', 'channel4', 'channel5', 'channel6', 'channel7', 'channel8','label'])
        # #     for item in train_2:
        # #         # writer.writerow([ item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7]])
        # #         for row in item:
        # #             writer.writerow([row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],'2'])
        # for item in train_10:
        #         #writer.writerow([ item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7]])
        #     for row in item:
        #         writer.writerow([row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],'10'])

                    
    train_x = []                                                        #train_x is the raw input data from the Myo.
    train_y = []                                                        #train_y is the label column. The "Classifier label" in the training dataset.
    for a in train_1:
        train_x.append(np.asarray(a))                                   #train_x is the raw input data from the Myo.
        train_y.append(1)                                               #train_y is the label column. The "Classifier label" in the training dataset.

    for a in train_2:
        train_x.append(np.asarray(a))                                   #train_x is the raw input data from the Myo.
        train_y.append(2)                                               #train_y is the label column. The "Classifier label" in the training dataset.

    for a in train_3:
        train_x.append(np.asarray(a))                                   #train_x is the raw input data from the Myo.
        train_y.append(3)                                               #train_y is the label column. The "Classifier label" in the training dataset.

    for a in train_4:
        train_x.append(np.asarray(a))                                   #train_x is the raw input data from the Myo.
        train_y.append(4)                                               #train_y is the label column. The "Classifier label" in the training dataset.

    for a in train_5:
        train_x.append(np.asarray(a))                                   #train_x is the raw input data from the Myo.
        train_y.append(5)                                               #train_y is the label column. The "Classifier label" in the training dataset.
        
    # for a in train_6:
    #     train_x.append(np.asarray(a))                                   #train_x is the raw input data from the Myo.
    #     train_y.append(6)                                               #train_y is the label column. The "Classifier label" in the training dataset.

    # for a in train_7:
    #     train_x.append(np.asarray(a))                                   #train_x is the raw input data from the Myo.
    #     train_y.append(7)                                               #train_y is the label column. The "Classifier label" in the training dataset.

    # for a in train_8:
    #     train_x.append(np.asarray(a))                                   #train_x is the raw input data from the Myo.
    #     train_y.append(8)                                               #train_y is the label column. The "Classifier label" in the training dataset.

    # for a in train_9:
    #     train_x.append(np.asarray(a))                                   #train_x is the raw input data from the Myo.
    #     train_y.append(9)                                               #train_y is the label column. The "Classifier label" in the training dataset.

    # for a in train_10:
    #     train_x.append(np.asarray(a))                                   #train_x is the raw input data from the Myo.
    #     train_y.append(10) 

    train_x_f = []                                                      #train_x_f is the featurized data. The "features" in the training dataset.簡單來說就是看到的[channel1~8]

    for a in train_x:
        x_f_h = []                                                      #x_f_h as the test dataset
        for b in range(0,8):
            x_f_h.append(rms(a[:, b]))                      
            x_f_h.append(iav(a[:, b]))
            x_f_h.append(ssi(a[:, b]))
            x_f_h.append(var(a[:, b]))
            #x_f_h.append(tm3(a[:, b]))
            x_f_h.append(wl(a[:, b]))
            x_f_h.append(aac(a[:, b]))
        train_x_f.append(x_f_h)                                         #train_x_f is the featurized data. The "features" in the training dataset.

# print(len(train_x_f), len(train_x))
    # clf1 = sklearn.ensemble.AdaBoostClassifier(n_estimators=30, learning_rate=1) #, random_state=np.random.randint(0,9))      #原作者寫的分類器(n_estimators=30代表最多30棵樹) 
    clf2 = sklearn.ensemble.RandomForestClassifier()                                                                            #原作者寫的分類器
    clf3 = sklearn.ensemble.RandomForestClassifier(n_estimators=30)                                                            #原作者寫的分類器
    clf4  = make_pipeline(SVC(kernel='linear',probability=True,decision_function_shape='0v0',random_state=123))
        #clf4參數以下介紹:
        # C: 目標函數的懲罰係數C，用來平衡分類間隔margin和錯分樣本的
        # kernel：參數選擇有RBF, Linear, Poly, Sigmoid, 默認的是"RBF"
        # probablity: 可能性估計是否使用(true or false)
        # random_state：用於概率估計的數據重排時的偽隨機數生成器的種子，又或是說決定資料的隨機方式
        # decision_function_shape='0v0' 为one v one(一對一)分类问题，即将类别两两之间进行划分，用二分类的方法模拟多分类的结果。

    # clf1.fit(train_x_f, train_y)
    clf2.fit(train_x_f, train_y)
    clf3.fit(train_x_f, train_y)
    clf4.fit(train_x_f, train_y)    #用訓練數據擬合分類器模型(train_x_f 和 train_y)  #train_x_f is the featurized data. The "features" in the training dataset. #train_y is the label column. The "Classifier label" in the training dataset.

    #X_iso = Isomap(n_neighbors=10).fit_transform(train_x_f)
                                            
    y_i = clf4.predict(train_x_f)              #train_x_f is the featurized data. The "features" in the training dataset.
    
    print("Collecting Complete!")

    print('SkLearn : ', metrics.accuracy_score(train_y, y_i))

    #print(train_x_f[0])                                                      #train_x_f is the featurized data. The "features" in the training dataset.

    X = []
    toEuler(listener.get_ori_data())
   
    
    f = open('output.csv', 'a',newline='')
        #configure writer to write standard csv file
    writer = csv.writer(f)
    writer.writerow(['channel1', 'channel2', 'channel3', 'channel4', 'channel5', 'channel6', 'channel7', 'channel8','label'])

    ff = open('label.csv','a',newline='')
    wwriter=csv.writer(ff)
    wwriter.writerow(['label'])
    while(1):
        # myo = feed.get_connected_devices()
        if len(X) > 20:
            test1.append(np.asarray(X))
            x_f_h = []                                                              #x_f_h as the test dataset
            X1 = np.asarray(X)
            x_f_h = []                                                              #x_f_h as the test dataset
            for b in range(0, 8):
                x_f_h.append(rms(X1[:, b]))
                x_f_h.append(iav(X1[:, b]))
                x_f_h.append(ssi(X1[:, b]))
                x_f_h.append(var(X1[:, b]))
                #x_f_h.append(tm3(X1[:, b]))
                x_f_h.append(wl(X1[:, b]))
                x_f_h.append(aac(X1[:, b]))
                #y_i = model.predict(np.column_stack(np.asarray(x_f_h)), verbose=0)
                #y_i_class = y_i.argmax(axis=-1)
            '''''
            a_i = 0
            y_i = y_i[0]
            max_var = max(y_i)
            for a in range(len(y_i)):
                if (y_i[a] == max_var):
                    a_i = a + 1
            '''''
            p2 = clf2.predict([x_f_h])       # Prediction or testing is then done using x_f_h as the test dataset. >> p2 = clf4.predict([x_f_h])   #原作者: p2 = clf2.predict([x_f_h])
            p3 = clf3.predict([x_f_h])       # Prediction or testing is then done using x_f_h as the test dataset. >> p4 = clf4.predict([x_f_h])   #原作者: p3 = clf3.predict([x_f_h])
            if p2 == p3:
                if p2[0] == 1: 
                    print('Pred --- ', ges[0])
                if p2[0] == 2:
                    print('Pred --- ', ges[1])
                if p2[0] == 3:
                    print('Pred --- ', ges[2])
                if p2[0] == 4:
                    print('Pred --- ', ges[3])
                if p2[0] == 5:
                    print('Pred --- ', ges[4])
                # if p2[0] == 6:
                #     print('Pred --- ', ges[5])
                # if p2[0] == 7:
                #     print('Pred --- ', ges[6])
                # if p2[0] == 8:
                #     print('Pred --- ', ges[7])
                # if p2[0] == 9:
                #     print('Pred --- ', ges[8])
                # if p2[0] == 10:
                #     print('Pred --- ', ges[9])

            for item in test1:
                    # writer.writerow([ row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],'1'])
                for row in item:
                    writer.writerow([row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],str(p2[0])])

            for item in test1:
                    # writer.writerow([ row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],'1'])
                for row in item:
                    wwriter.writerow([str(p2[0])])

            X = []       
    sleep(1)

# Conclusion:
# train_x is the raw input data from the Myo. 
# train_x_f is the featured data. The "features" in the training dataset.
# train_y is the label column. The "Classifier label" in the training dataset. 
# Once the classifier is trained on (train_x_f, train_y), I do the testing, or as I called it, prediction, in the last while loop.>> clf4.fit(train_x_f, train_y)
# X is the raw input data from the Myo. Once there are more than 20 rows appended to X, it is featurized and called x_f_h.
# Prediction or testing is then done using x_f_h as the test dataset. >> p2 = clf2.predict([x_f_h])




