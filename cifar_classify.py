import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Getting the CIFAR-10 data
CIFAR_DIREC = 'cifar-10-python/cifar-10-batches-py/'
# Unpickle the data using cPickle Library
def unpickle(file):
    import pickle
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    return dict
# List of names of all the files
all_files = ['batches.meta', 'data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']
# Seperating and assigning the data 
all_indices = [i for i in range(7)]

for i,j in zip(all_indices, all_files):
    all_indices[i] = unpickle(CIFAR_DIREC+j)

batches_meta = all_indices[0]
data_batch_1 = all_indices[1]
data_batch_2 = all_indices[2]
data_batch_3 = all_indices[3]
data_batch_4 = all_indices[4]
data_batch_5 = all_indices[5]
test_batch = all_indices[6]

# Create a one hot encoder function

def one_hot_encode(vec, classes=10):
    length = len(vec)
    matrix = np.zeros((length, classes))
    matrix[range(length), vec] = 1
    return matrix
# Create a CIFAR Helper Class

class CIFAR_Helper():
    # Initializer
    def __init__(self):
        # Iteration Count Variable
        self.i = 0
        # Put all training batches into a single list
        self.all_training_batches = [data_batch_1, data_batch_2,data_batch_3,data_batch_4,data_batch_5]
        # Put the test batch in a seperate list
        self.all_test_batches = [test_batch]

        # Create some useful variables
        self.training_images = None
        self.training_labels = None
        self.test_images = None
        self.test_labels = None

    def handle_images(self):
        print("Setting Up Traning Images .... \n")
        # Setting up training images
        self.training_images = np.vstack([data[b'data'] for data in self.all_training_batches])
        train_len = len(self.training_images)
        # Reshaping and Normalizing the training images
        self.training_images = self.training_images.reshape(train_len,3,32,32).transpose(0,2,3,1).astype('uint8')/255
        self.training_labels = one_hot_encode(np.hstack([data[b'labels'] for data in self.all_training_batches]))
        print('Done!!\n\n')

        print("Setting Up Test Images .... \n")
        # Setting up testing images
        self.test_images = np.vstack([data[b'data'] for data in self.all_test_batches])
        test_len = len(self.test_images)
        # Reshaping and Normalizing test images
        self.test_images = self.test_images.reshape(test_len,3,32,32).transpose(0,2,3,1).astype('uint8')/255
        self.test_labels = one_hot_encode(np.hstack([data[b'labels'] for data in self.all_test_batches]))
        print('Done!!\n\n')

    # Helper function to get the next batch 
    def get_next_batch(self, batch_size):
        # Load the next batch
        x = self.training_images[self.i:self.i+batch_size].reshape(batch_size,32,32,3).astype('uint8')
        y = self.training_labels[self.i:self.i+batch_size]
        # Update the value of i
        self.i = (self.i + batch_size)%len(self.training_images)
        return x, y

# Creating a CIFARHelper Instance and setting up images
ch = CIFAR_Helper()
ch.handle_images()

################ CREATING THE MODEL ##################

# Making Placeholders for the input variables and hyperparameters

x = tf.placeholder(tf.float32, shape=[None, 32,32,3])
y = tf.placeholder(tf.float32, shape=[None, 10])
hold_prob = tf.placeholder(tf.float32)

# Helper functions for the various layers

def init_weights(shape):
    weights = tf.truncated_normal(shape, stddev=0.1)
    return weights

def init_bias(shape):
    bias = tf.constant(0.2, dtype=tf.float32, shape=shape)
    return tf.Variable(bias)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=(1,1,1,1), padding='SAME')

def maxpool(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def convolutional_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x, W)+b)

def normal_full_layer(x, size):
    input_size = int(x.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_weights([size])
    return tf.matmul(x,W)+b

# Define the Model
conv_1 = convolutional_layer(x, shape=[4,4,3,32])
max_pool_conv_1 = maxpool(conv_1)

conv_2 = convolutional_layer(max_pool_conv_1, shape=[4,4,32,64])
max_pool_conv_2 = maxpool(conv_2)

# Flatten the pooled layers
conv_2_flat = tf.reshape(max_pool_conv_2, [-1,8*8*64])

# Adding fully connected layers
full_layer_1 = tf.nn.relu(normal_full_layer(conv_2_flat, 4096))
full_layer_2 = tf.nn.relu(normal_full_layer(full_layer_1, 1024))

# Adding dropout
full_layer_dropout = tf.nn.dropout(full_layer_2, keep_prob = hold_prob)

# Storing predicted values        
y_pred = normal_full_layer(full_layer_dropout, 10)

######### LOSS FUNCTION ###########

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y))

# Using Adam Optimizer Instance
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

train = optimizer.minimize(cross_entropy)

# Initializer Variables
init = tf.global_variables_initializer()

epochs=1000

# Running Tensorflow Session
with tf.Session() as sess:
    sess.run(init)

    for i in range(epochs):
        # Get the next batch
        batch = ch.get_next_batch(100)
        # Run the optimizer to reduce loss
        sess.run(train, feed_dict={x:batch[0], y:batch[1], hold_prob:0.5})
        # Log the Accuracy after every Hundred steps
        if(i%100)==0:
            print("Epoch " + str(i+1) + "\n")
            accuracy = tf.equal(tf.argmax(y_pred,1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))
            print(sess.run(accuracy, feed_dict={x:ch.test_images, y:ch.test_labels, hold_prob:1.0}))