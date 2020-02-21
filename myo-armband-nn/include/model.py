# This is Myo Armband via neural network model 
# CopyRight @ wmlab & NCU 2019 Apr

import tensorflow as tf

def model(_NUM_CLASSES = 5): #_NUM_CLASSES:结果是要得到一个几分类的任务，total classes(0-4)

    with tf.name_scope('data'):
        # 输入和输出
        x = tf.placeholder(tf.float32, shape=[None, 64], name='Input')
        y = tf.placeholder(tf.float32, shape=[None, _NUM_CLASSES], name='Output')

    # Store layers weight & bias
    # tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)函数用于从服从指定正太分布的数值中取出指定个数的值。
    # shape: 输出张量的形状，必选
    # mean: 正态分布的均值，默认为0 stddev: 正态分布的标准差，默认为1.0 dtype: 输出的类型，默认为tf.float32 seed: 随机数种子，是一个整数，当设置之后，每次生成的随机数都一样 name: 操作的名称
    # 变量(Variable):当创建一个变量时，你将一个张量作为初始值传入构造函数Variable()。TensorFlow提供一系列操作初始化张量，初始值是常量或随机值。
    
    #stddev = 0.1 #自己加的
    #定义权重
    weights = {
        'h1': tf.Variable(tf.random_normal([64, 528])),                 #tf.random_normal([n_input,n_hidden_1])，所以input=64,hidden_1=528
        'h2': tf.Variable(tf.random_normal([528, 786])),
        'h3': tf.Variable(tf.random_normal([786, 1248])),
        'out': tf.Variable(tf.random_normal([1248, _NUM_CLASSES]))
        # 'h1': tf.Variable(tf.random_normal([64, 128])),                 
        # 'h2': tf.Variable(tf.random_normal([128, 128])),
        # 'h3': tf.Variable(tf.random_normal([128, 128])),
        # 'h4': tf.Variable(tf.random_normal([128, 64])),                 
        # 'h5': tf.Variable(tf.random_normal([64, 32])),
        # 'h6': tf.Variable(tf.random_normal([32, 16])),
        # 'out': tf.Variable(tf.random_normal([16, _NUM_CLASSES]))      
    }
    #定义偏置
    biases = {
        'b1': tf.Variable(tf.random_normal([528])),          #1st layer neurons features have 528
        'b2': tf.Variable(tf.random_normal([786])),          #2nd layer neurons features have 786
        'b3': tf.Variable(tf.random_normal([1248])),         #3rd layer neurons features have 1248
        'out': tf.Variable(tf.random_normal([_NUM_CLASSES]))
        # 'b1': tf.Variable(tf.random_normal([128])),            #1st layer neurons features have 128
        # 'b2': tf.Variable(tf.random_normal([128])),            #2nd layer neurons features have 128
        # 'b3': tf.Variable(tf.random_normal([128])),            #3rd layer neurons features have 128
        # 'b4': tf.Variable(tf.random_normal([64])),             #4th layer neurons features have 64
        # 'b5': tf.Variable(tf.random_normal([32])),             #5th layer neurons features have 32
        # 'b6': tf.Variable(tf.random_normal([16])),             #6th layer neurons features have 16
        # 'out': tf.Variable(tf.random_normal([_NUM_CLASSES]))
    }
    print("神經網路模型已完成載入...")

    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])   #tf.matmul（）将矩阵a乘以矩阵b，生成a * b
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)
    
    # layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    # layer_4 = tf.nn.relu(layer_4)

    # layer_5 = tf.add(tf.matmul(layer_4, weights['h5']), biases['b5'])
    # layer_5 = tf.nn.relu(layer_5)

    # layer_6 = tf.add(tf.matmul(layer_5, weights['h6']), biases['b6'])
    # layer_6 = tf.nn.relu(layer_6)

    layer_3 = tf.nn.dropout(layer_3, 0.5)
    #layer_6 = tf.nn.dropout(layer_6, 0.5) #tf.nn.dropout()是tensorflow里面为了防止或减轻过拟合而使用的函数，随机的拿掉网络中的部分神经元，从而减小对W权重的依赖，以达到减小过拟合的效果。
    
    # tf.nn.dropout(x, keep_prob, noise_shape=None, seed=None, name=None) #x：输入，keep_prob：保留比例，取值 (0,1] 。每一个参数都将按这个比例随机变更，当keep_prob=1的时候，相当于100%保留，也就是dropout没有起作用。
    # output = tf.add(tf.matmul(layer_3, weights['out']), biases['out'], name="output")
    output = tf.add(tf.matmul(layer_3, weights['out']), biases['out'], name="output")

    global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)  #global_step在滑动平均、优化器、指数衰减学习率等方面都有用到，这个变量的实际意义非常好理解：代表全局步数，比如在多少步该进行什么操作，现在神经网络训练到多少轮等等，类似于一个钟表。
    y_pred_cls = tf.argmax(output, dimension=1)

    

    return x, y, output, global_step, y_pred_cls