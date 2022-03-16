import tensorflow as tf
import numpy as np
import time
import random
import os

import util
import sys

ITER = int(sys.argv[1])
USE_OOO = int(sys.argv[2])

tf.compat.v1.disable_eager_execution()

SEED = 1
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

BATCH_SIZE = 1024
INPUT_SIZE = 1024
SIZE = 1024
num_classes = 1024

def build_model( X, scope_name ):
  with tf.name_scope(scope_name):
    with  tf.compat.v1.variable_scope("BLOCK_1", reuse=tf.compat.v1.AUTO_REUSE):
      with tf.device( '/GPU:0' ):
        w1 = tf.compat.v1.get_variable( "W1", [INPUT_SIZE, SIZE], initializer=tf.random_normal_initializer(stddev=0.01) )
        fc1 = tf.matmul( X, w1 )
        fc1 = tf.nn.relu( fc1 )
        w2 = tf.compat.v1.get_variable( "W2", [INPUT_SIZE, SIZE], initializer=tf.random_normal_initializer(stddev=0.01) )
        fc2 = tf.matmul( fc1, w2 )
        fc2 = tf.nn.relu( fc2 )
    
    with  tf.compat.v1.variable_scope("BLOCK_2", reuse=tf.compat.v1.AUTO_REUSE):
      with tf.device( '/GPU:1' ):
        w3 = tf.compat.v1.get_variable( "W3", [SIZE, num_classes], initializer=tf.random_normal_initializer(stddev=0.01) )
        fc3 = tf.matmul( fc2, w3 )
        fc3 = tf.nn.relu( fc3 )
        w4 = tf.compat.v1.get_variable( "W4", [SIZE, num_classes], initializer=tf.random_normal_initializer(stddev=0.01) )
        fc4 = tf.matmul( fc3, w4 )
        fc4 = tf.nn.relu( fc4 )
  
  logit = fc4
  return logit

DUMMY_X = np.random.normal(0,1,(BATCH_SIZE, INPUT_SIZE))
X = tf.compat.v1.placeholder(tf.float32, (BATCH_SIZE, INPUT_SIZE), name = 'INPUT_IMG')
DUMMY_Y = np.random.normal(0,1,(BATCH_SIZE, num_classes))
Y = tf.compat.v1.placeholder(tf.float32, (BATCH_SIZE, num_classes), name = 'INPUT_IMG')

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.0001)


with tf.device( '/GPU:1' ):
    logits = build_model( X, "MODEL_1" )
    cost = tf.reduce_mean(
        input_tensor = tf.compat.v1.losses.softmax_cross_entropy( Y, logits )
    )

gvs = optimizer.compute_gradients(cost, colocate_gradients_with_ops=True)
train_ops = optimizer.apply_gradients(gvs)

if( USE_OOO ):
    util.schedule_ooo_backpropagation( tf.compat.v1.get_default_graph() )

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())


for step in range(100):
  sess.run([train_ops], feed_dict={X:DUMMY_X, Y:DUMMY_Y}) 
  if( step+1 == ITER ):
      print( "STEP: ", step+1 )
      logit_val = sess.run([logits], feed_dict={X:DUMMY_X, Y:DUMMY_Y})
      print(logit_val)
      break

