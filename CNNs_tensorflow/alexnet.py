from datetime import datetime
import math
import time
import tensorflow as tf
batch_size=32
num_batch=100

def print_activation(t):
    print(t.op.name,t.get_shape().as_list())

def inference(images):
    parameters=[]
    with tf.name_scope('conv1') as scope:
        kernel=tf.Variable(tf.truncated_normal([11,11,3,64],dtype=tf.float32,stddev=1e-1),name='weights')
        conv=tf.nn.conv2d(images,kernel,[1,4,4,1],padding='SAME')
        biases=tf.Variable(tf.constant(0.0,shape=[64],dtype=tf.float32),trainable=True,name='biases')
        bias=tf.nn.bias_add(conv,biases)
        conv1=tf.nn.relu(bias,name=scope)
        print_activation(conv1)
        parameters+=[kernel,biases]
        pool1=tf.nn.max_pool(conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name='pool1')
        print_activation(pool1)

    with tf.name_scope('conv2') as scope:
        kernel=tf.Variable(tf.truncated_normal([5,5,64,192],dtype=tf.float32,stddev=1e-1),name='weights')
        conv=tf.nn.conv2d(pool1,kernel,[1,1,1,1],padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activation(conv2)
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2,1 ], padding='VALID', name='pool2')
        print_activation(pool2)

    with tf.name_scope('conv3') as scope:
        kernel=tf.Variable(tf.truncated_normal([3,3,192,384],dtype=tf.float32,stddev=1e-1),name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        parameters+=[kernel, biases]
        print_activation(conv3)

    with tf.name_scope('conv4') as scope:
        kernel=tf.Variable(tf.truncated_normal([3,3,384,256],dtype=tf.float32,stddev=1e-1),name='weights')
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        parameters+=[kernel, biases]
        print_activation(conv4)

    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activation(conv5)

    pool5=tf.nn.max_pool(conv5,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name='poll5')
    print_activation(pool5)
    return pool5,parameters

def time_tensorflow(session,target,info_string):
    num_step_warming_up=10
    total_duration=0.0
    total_duration_squared=0.0
    for i in range (num_batch+num_step_warming_up):
        start_time=time.time()
        _=session.run(target)
        duration=time.time()-start_time
        if i<num_step_warming_up:
            if not i%10:
                print('%s:step:%d,duration=%.3f'%(datetime.now(),i-num_step_warming_up,duration))
            total_duration+=duration
            total_duration_squared+=duration*duration
            mn=total_duration/num_batch
            vr=total_duration_squared/num_batch-mn*mn
            sd=math.sqrt(vr)
            print ('%s:%s across %d steps,%.3f+/-%.3f sec/batch'%(datetime.now(),info_string,num_batch,mn,sd))



def run_benchmark():
    with tf.Graph().as_default():
        image_size=224
        images=tf.Variable(tf.random_normal([batch_size,image_size,image_size,3],dtype=tf.float32,stddev=1e-1))
        pool5,parameters=inference(images)
        init=tf.global_variables_initializer()
        sess=tf.Session()
        sess.run(init)
        time_tensorflow(sess,pool5,'forward')
        objective=tf.nn.l2_loss(pool5)
        grad=tf.gradients(objective,parameters)
        time_tensorflow(sess,grad,'forward-barkwork')

run_benchmark()