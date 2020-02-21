# Myo Armband can collect EmgData, 5 gestures (spreadfinger,wavein,waveout,fist,rest) also can convert to CSV file .
# CopyRight @ wmlab & NCU 2019 NＯＶ
from __future__ import print_function 
from myo import init, Hub, DeviceListener, StreamEmg, ApiDeviceListener
from time import sleep 
import numpy as np
import threading
import collections
import math
import myo               
import csv    
import pandas as pd
import time
from enum import Enum
##===============Myo Armband的Feature Extraction===============##
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def rms(array):                    #Root mean square
    n = len(array)
    sum = 0
    for a in array:
        sum =+ a*a
    return np.sqrt((1/float(n))*sum)

# def iav(array):                    #Integrated absolute value
#     sum = 0
#     for a in array:
#         sum += np.abs(a)
#     return sum

# def ssi(array):                    #Simple Square Integral    
#     sum = 0
#     for a in array:
#         sum += a*a
#     return sum

# def var(array):                    #Variance
#     n = len(array)
#     sum = 0
#     for a in array:
#         sum += a*a
#     return ((1/float(n-1))*sum)

# def tm3(array):
#     n = len(array)
#     print('n : ', n)
#     sum = 0
#     for a in array:
#         sum =+ a*a*a
#     return np.power((1/float(n))*sum,1/float(3))

# def wl(array):                      #window length:It has amplitude related features, which represents the cumulative length of the sEMG waveform over the time segment.
#     sum = 0
#     for a in range(0,len(array)-1):
#         sum =+ array[a+1] - array[a]
#     return sum
    
# def aac(array):                     #Average Amplitude Change: measures average of the amplitude change in signal;
#     n = len(array)
#     sum = 0
#     for a in range(0,n-1):
#         sum =+ array[0+1] - array[0]
#     return sum/float(n)

def featurize(array):
    n = []
    for a in array:
        n.append(rms(a))
    return n
##===============Myo Armband的Feature Extraction===============##

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

    #ges1 = ['Rock', 'Paper', 'Scissors', 'Lizard', 'Spock']
    #ges2 = ['channel 1', 'channel 2', 'channel 3', 'channel 4', 'channel 5']
    ges3 = ['手掌張開', '手掌往右擺', '手掌往左擺', '握拳', '放鬆']
    ges = ges3

    #參考資料:https://blog.csdn.net/hdandan2015/article/details/78719915
    now = time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime(time.time()))
    fname="C:/Users/ROG/Desktop/myo-armband-nn-master/data/Kaohsiung Chang Gung Memorial Hospital Report/"+now+r".csv" 
    csvFile = open(fname,'w',newline='')
    writer = csv.writer(csvFile)
    writer.writerow(['emg','gesture'])
    # f = open('20190823traindata.csv', 'a',newline='')
    # writer = csv.writer(f)
    # writer.writerow(['emg','gesture'])

    for a in range(1,2):   #次數

        print("\nGesture -- ", ges[0]," : Ready?")
        print("\n======手勢練習開始倒數計時程序======")
        for x in range(5,-1,-1):                                    #range(开始,结束,步长)
            mystr = "倒數計時"+str(x)+"秒"
            print(mystr,end = "")
            print("\b"*(len(mystr)*2),end = "",flush = True)
            time.sleep(2)
        print("\n======開始蒐集您的手臂肌肉訊號...(請勿移動)======")
        #input("Press Enter to continue...") #把按鈕Enter鍵取消
        start = time.process_time()                         #開始測量時間，time.process_time()不會讓計算結果受到其他程式的影響，會以 CPU time 來計算
        X = []
        while(1):
            if len(X) >=64 :
                train_1.append(np.asarray(X))
                X = []
                if len(train_1) > a*req_iter:
                    break 
            
        for item in train_1:
                # writer.writerow([ row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],'1'])
            for row in item:
                srow=str(row)
                srow=srow.strip('[]')+' '+srow.strip('[]')+' '+srow.strip('[]')+' '+srow.strip('[]')+' '+srow.strip('[]')+' '+srow.strip('[]')+' '+srow.strip('[]')+' '+srow.strip('[]')
                srow=srow.replace(' ',';')
                writer.writerow([srow,'1'])
        end = time.process_time()                           #結束測量時間
        print('-------------------------------------------')
        print("蒐集手掌張開資料執行時間：%f 秒" % (end - start))      #輸出測量Loop測量時間結果
        print('-------------------------------------------')
        print("\n======準備下一個階段的手勢======")
                #print(srow)
        print('-------------------------------------------')
        print("手掌張開的肌肉電資料長度 : ",item.shape)
        print('-------------------------------------------')

        print("\nGesture -- ", ges[1]," : Ready?")
        print("\n======手勢練習開始倒數計時程序======")
        for x in range(5,-1,-1):                                    #range(开始,结束,步长)
            mystr = "倒數計時"+str(x)+"秒"
            print(mystr,end = "")
            print("\b"*(len(mystr)*2),end = "",flush = True)
            time.sleep(2)
        print("\n======開始蒐集您的手臂肌肉訊號...(請勿移動)======")
        #input("Press Enter to continue...")
        start = time.process_time()         # 開始測量，time.process_time()不會讓計算結果受到其他程式的影響，會以 CPU time 來計算
        X = []
        while(1):
            if len(X) >= 64:
                # print(X[-1])
                train_2.append(np.asarray(X))
                X = []
                if len(train_2) > a*req_iter:
                    break
        
        for item in train_2:
                #writer.writerow([ item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7]])
            for row in item:
                srow=str(row)
                srow=srow.strip('[]')+' '+srow.strip('[]')+' '+srow.strip('[]')+' '+srow.strip('[]')+' '+srow.strip('[]')+' '+srow.strip('[]')+' '+srow.strip('[]')+' '+srow.strip('[]')
                srow=srow.replace(' ',';')
                writer.writerow([srow,'2'])
        end = time.process_time()                  # 結束測量
        print('-------------------------------------------')
        print("蒐集手掌往右擺資料執行時間：%f 秒" % (end - start))    # 輸出結果
        print('-------------------------------------------')
        print("\n======準備下一個階段的手勢======")
                #print(srow)
        print('-------------------------------------------')
        print("手掌往右擺的肌肉電資料長度 : ",item.shape)
        print('-------------------------------------------')
                
        print("\nGesture -- ", ges[2]," : Ready?")
        print("\n======手勢練習開始倒數計時程序======")
        for x in range(5,-1,-1):                                    #range(开始,结束,步长)
            mystr = "倒數計時"+str(x)+"秒"
            print(mystr,end = "")
            print("\b"*(len(mystr)*2),end = "",flush = True)
            time.sleep(2)
        print("\n======開始蒐集您的手臂肌肉訊號...(請勿移動)======")
        #input("Press Enter to continue...")
        start = time.process_time()         # 開始測量，time.process_time()不會讓計算結果受到其他程式的影響，會以 CPU time 來計算
        X = []
        while(1):
            if len(X) >= 64:
                # print(X[-1])
                train_3.append(np.asarray(X))
                X = []
                if len(train_3) > a*req_iter:
                    break
        
        for item in train_3:
                #writer.writerow([ item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7]])
            for row in item:
                srow=str(row)
                srow=srow.strip('[]')+' '+srow.strip('[]')+' '+srow.strip('[]')+' '+srow.strip('[]')+' '+srow.strip('[]')+' '+srow.strip('[]')+' '+srow.strip('[]')+' '+srow.strip('[]')
                srow=srow.replace(' ',';')
                writer.writerow([srow,'3'])
                #print(srow)
        end = time.process_time()                               #結束測量
        print('-------------------------------------------')
        print("蒐集手掌往左擺資料執行時間：%f 秒" % (end - start))          #輸出結果
        print('-------------------------------------------')
        print("\n======準備下一個階段的手勢======")
        print('-------------------------------------------')
        print("手掌往左擺的肌肉電資料長度 : ",item.shape)
        print('-------------------------------------------')

        print("\nGesture -- ", ges[3]," : Ready?")
        print("\n======手勢練習開始倒數計時程序======")
        for x in range(5,-1,-1):                                    #range(开始,结束,步长)
            mystr = "倒數計時"+str(x)+"秒"
            print(mystr,end = "")
            print("\b"*(len(mystr)*2),end = "",flush = True)
            time.sleep(2)
        print("\n======開始蒐集您的手臂肌肉訊號...(請勿移動)======")
        #input("Press Enter to continue...")
        start = time.process_time()         # 開始測量，time.process_time()不會讓計算結果受到其他程式的影響，會以 CPU time 來計算
        X = []
        while(1):
            if len(X) >= 64:
                # print(X[-1])
                train_4.append(np.asarray(X))
                X = []
                if len(train_4) > a*req_iter:
                    break
       
        for item in train_4:
                #writer.writerow([ item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7]])
            for row in item:
                srow=str(row)
                srow=srow.strip('[]')+' '+srow.strip('[]')+' '+srow.strip('[]')+' '+srow.strip('[]')+' '+srow.strip('[]')+' '+srow.strip('[]')+' '+srow.strip('[]')+' '+srow.strip('[]')
                srow=srow.replace(' ',';')
                writer.writerow([srow,'4'])
                #print(srow)
        end = time.process_time()               # 結束測量
        print('-------------------------------------------')
        print("蒐集手掌握拳資料執行時間：%f 秒" % (end - start)) # 輸出結果
        print('-------------------------------------------')
        print("\n======準備下一個階段的手勢======")
        print('-------------------------------------------')
        print("手掌握拳的肌肉電資料長度 : ",item.shape)
        print('-------------------------------------------')

        print("\nGesture -- ", ges[4]," : Ready?")
        print("\n======手勢練習開始倒數計時程序======")
        print('\n')
        for x in range(5,-1,-1):                                    #range(开始,结束,步长)
            mystr = "倒數計時"+str(x)+"秒"
            print(mystr,end = "")
            print("\b"*(len(mystr)*2),end = "",flush = True)
            time.sleep(2)
        print("\n======開始蒐集您的手臂肌肉訊號...(請勿移動)======")
        #input("Press Enter to continue...")
        start = time.process_time()                         # 開始測量，time.process_time()不會讓計算結果受到其他程式的影響，會以 CPU time 來計算
        X = []
        while(1):
            if len(X) >= 64:
                # print(X[-1])
                train_5.append(np.asarray(X))
                X = []
                if len(train_5) > a*req_iter:
                    break 
        
        for item in train_5:
                #writer.writerow([ item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7]])
            for row in item:
                srow=str(row)
                srow=srow.strip('[]')+' '+srow.strip('[]')+' '+srow.strip('[]')+' '+srow.strip('[]')+' '+srow.strip('[]')+' '+srow.strip('[]')+' '+srow.strip('[]')+' '+srow.strip('[]')
                srow=srow.replace(' ',';')
                writer.writerow([srow,'5'])
                #print(srow)
        end = time.process_time()                             # 結束測量
        print('-------------------------------------------')
        print("蒐集手掌放鬆資料執行時間：%f 秒" % (end - start))        # 輸出結果
        print('-------------------------------------------')
        print("手掌放鬆的肌肉電資料長度 : ",item.shape)
        print('-------------------------------------------')
        print("Storing your muscle signal...Please waiting...")
        

train_x = []
train_y = []
    
for a in train_1:
    train_x.append(np.asarray(a))
    #print(sum(rms(train_1)))
    train_y.append(1)

for a in train_2:
    train_x.append(np.asarray(a))
    train_y.append(2)

for a in train_3:
    train_x.append(np.asarray(a))
    train_y.append(3)

for a in train_4:
    train_x.append(np.asarray(a))
    train_y.append(4)

for a in train_5:
    train_x.append(np.asarray(a))
    train_y.append(5)

train_x_f = []

for a in train_x:
    x_f_h = []
    for b in range(0,8):
        x_f_h.append(rms(a[:, b]))
        #x_f_h.append(iav(a[:, b]))
        #x_f_h.append(ssi(a[:, b]))
        #x_f_h.append(var(a[:, b]))
        #x_f_h.append(tm3(a[:, b]))
        #x_f_h.append(wl(a[:, b]))
        #x_f_h.append(aac(a[:, b]))
        #x_f_h.append(IEMG(a[:, b]))
    train_x_f.append(x_f_h)

#print(train_x_f[0]) 
print('-------------------------------')
print("Collecting Data Complete ! 已經完成蒐集好您的肌肉電訊號!")
print('-------------------------------')
#print('Window Length Value(WL):',wl(a))
#print("=================================")
# print('Root mean square Value(RMS):',sum(featurize(a)))
# print("=================================")
#print('Integrated absolute value(IEMG):',IEMG(a))
#print("=================================")
# print(a)
# print("=================================")
# print(sum(rms(a)))
# print("=================================")

# Conclusion:
# train_x is the raw input data from the Myo. 
# train_x_f is the featured data. The "features" in the training dataset.
# train_y is the label column. The "Classifier label" in the training dataset. 
# Once the classifier is trained on (train_x_f, train_y), I do the testing, or as I called it, prediction, in the last while loop.>> clf4.fit(train_x_f, train_y)
# X is the raw input data from the Myo. Once there are more than 20 rows appended to X, it is featurized and called x_f_h.
# Prediction or testing is then done using x_f_h as the test dataset. >> p2 = clf2.predict([x_f_h])