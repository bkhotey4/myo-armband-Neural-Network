# This is Myo Armband via neural network model 
# CopyRight @ wmlab & NCU 2019 Apr

import numpy as np
import pandas as pd
import keras.utils
import csv
import math 
from math import sqrt
#import os.path
import os

def get_data_set(name="train"):

    if name is "train":
        for info in os.listdir('C:/Users/ROG/Desktop/myo-armband-nn-master/data/Kaohsiung Chang Gung Memorial Hospital Report/Train_Dataset/'):           
            domain = os.path.abspath(r'C:/Users/ROG/Desktop/myo-armband-nn-master/data/Kaohsiung Chang Gung Memorial Hospital Report/Train_Dataset/')     #獲取文件夹的路径
            info = os.path.join(domain,info) #將路径與文件名结合起来就是每個文件的完整路径
            df=pd.read_csv(info)
            print('Loading...csvfile...List:',info)
            #print('\n')
            x_train = df['emg']
            print(x_train.shape)
            y_train = df['gesture']
            x_train_length = len(df['emg'])
            x_train_list = []
            for i in range(0, x_train_length):
                x_train_string = x_train[i].replace(';', ' ')
                temp = list(map(int, x_train_string.split()))
                x_train_list.append(temp)
            #print(x_train_length)
            #print(type(x_train_list[4180][0]))
            x_train_array = np.array(x_train_list)
            #print(x_train_array)
            #print(x_train_array.shape)  #4181, 64
        
            #y_train 改成 array
            y_train_array = np.array(y_train)
            #print(y_train_array)
            #print(np.array(y_train).shape) #4181, 
            #修改y_trainlabel方式
            y_train_array = y_train_array-1
            #使用onehotencode方式編碼y_train
            y_train_onehot = keras.utils.to_categorical(y_train_array, num_classes=5)
            #print(x_train_array)
            #print(y_train_array)
            #print(len(y_train_array))

            # print("-------------------------------")
            # #資料個數
            # count = len(x_train)
            # print("資料行數Count = " ,count)
            # print("-------------------------------")
            #RMS(Root Mean Square)
            #rms_value=np.sqrt(np.mean(x_train_array)**2)    
            # print("Root Mean Square Value =",rms_value)
            # print("-------------------------------")
            x = x_train_array
            y = y_train_onehot        
        return x, y

    elif name is "test":
        #==================原本================== 
    #     df=pd.read_csv('./data/Kaohsiung Chang Gung Memorial Hospital Report/2019-09-17-15-48-10.csv',header=0,sep=',')
    #     x_train = df['emg']
    #     y_train = df['gesture']
    #     x_train_length = len(df['emg'])     #train資料集長度  
    #     x_train_list = []
        
    #     for i in range(0, x_train_length):
    #         x_train_string = x_train[i].replace(';', ' ')
    #         temp = list(map(int, x_train_string.split()))
    #         x_train_list.append(temp)
    
    #     #print(x_train_length)
    #     #print(type(x_train_list[4180][0]))
    #     x_train_array = np.array(x_train_list)

    #     #print(x_train_array.shape)  #4181, 64
        
    #     #y_train 改成 array
    #     y_train_array = np.array(y_train)
    #     print(np.array(y_train).shape) #4181, 
    #     #修改y_trainlabel方式
    #     y_train_array = y_train_array-1
    #     #使用onehotencode方式編碼y_train
    #     y_train_onehot = keras.utils.to_categorical(y_train_array, num_classes=5)

    #     print(x_train_array)
    #     print(y_train_onehot)

    #     x = x_train_array
    #     y = y_train_onehot
        
    # return x, y
    #==================原本================== 

#==================讀取資料夾底下所有的CSV==================   
        for info in os.listdir('C:/Users/ROG/Desktop/myo-armband-nn-master/data/Kaohsiung Chang Gung Memorial Hospital Report/'):           
            domain = os.path.abspath(r'C:/Users/ROG/Desktop/myo-armband-nn-master/data/Kaohsiung Chang Gung Memorial Hospital Report/')     #獲取文件夹的路径
            info = os.path.join(domain,info) #將路径與文件名结合起来就是每個文件的完整路径
            df=pd.read_csv(info)
            print('載入CSV檔案...:',info)
            x_train = df['emg']
            #print(x_train.shape)
            y_train = df['gesture']
            x_train_length = len(df['emg'])
            x_train_list = []
            for i in range(0, x_train_length):
                x_train_string = x_train[i].replace(';', ' ')
                temp = list(map(int, x_train_string.split()))
                x_train_list.append(temp)
            #print(x_train_length)
            #print(type(x_train_list[4180][0]))
            x_train_array = np.array(x_train_list)
            #print(x_train_array)
            #print(x_train_array.shape)  #4181, 64
        
            #y_train 改成 array
            y_train_array = np.array(y_train)
            #print(y_train_array)
            #print(np.array(y_train).shape) #4181, 
            #修改y_trainlabel方式
            y_train_array = y_train_array-1
            #使用onehotencode方式編碼y_train
            y_train_onehot = keras.utils.to_categorical(y_train_array, num_classes=5)
            #print(x_train_array)
            #print(y_train_array)
            #print(len(y_train_array))


            x = x_train_array
            y = y_train_onehot        
        return x, y
#==================讀取資料夾底下所有的CSV==================

if __name__ == "__main__":
    #print('asafd')
    get_data_set('train')
    #get_data_set('test')

