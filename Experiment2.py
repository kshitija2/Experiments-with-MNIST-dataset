# Training set: Took the first 40,000, the next 40,000 starting from the 10,001st, 
# and the last 40,000 of the original training images, lined them up in three rows, 
# and then concatenate three images in each column to make 40,000 three digit images; 
# Testing set: concatenated in the same way the first 8,000, the next 8,000 starting from the 1,001st, 
# and the last 9,000 of the original testing images. 
# The output became ordered three digits that may be labeled as numbers between 0 and 999.
# Results:
# Final Training Accuracy: 0.953725
# Final Testing Accuracy: 0.86875

# Import MNIST data
import math
import numpy as np
import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("C:/Users/krta225/Downloads/mnistData", one_hot=True)
train_images=np.concatenate((mnist.train.images,mnist.validation.images))
train_labels=np.concatenate((mnist.train.labels,mnist.validation.labels))
test_images=mnist.test.images
test_labels=mnist.test.labels
images1=train_images[0:40000]
images2=train_images[10000:50000]
images3=train_images[20000:60000]
new_train_images=[]
for i in range(len(images1)):
 temp=np.concatenate((images1[i],images2[i],images3[i]))
 new_train_images.append(temp)
new_train_images=np.array(new_train_images)
labels1=train_labels[0:40000]
labels2=train_labels[10000:50000]
labels3=train_labels[20000:60000]
new_train_labels=[]
for i in range(len(labels1)):
 temp=np.array([])
 for j in range(1000):
 temp=np.append(temp,0)
 index1 = np.where(labels1[i]==1)
 index2 = np.where(labels2[i]==1)
 index3 = np.where(labels3[i]==1)
 s1=int(index1[0])
 s2=int(index2[0])
 s3=int(index3[0])
 s4=str(s1)
 s5=str(s2)
 s6=str(s3)
 s7=s4+s5+s6
 s8=int(s7)
 temp[s8]=1
 new_train_labels.append(temp)
new_train_labels=np.array(new_train_labels)
test1=test_images[0:8000]
test2=test_images[1000:9000]
test3=test_images[2000:10000]
new_test_images=[]
for i in range(len(test1)):
 temp=np.concatenate((test1[i],test2[i],test3[i]))
 new_test_images.append(temp)
new_test_images=np.array(new_test_images)
test1_labels=test_labels[0:8000]
test2_labels=test_labels[1000:9000]
test3_labels=test_labels[2000:10000]
new_test_labels=[]
for i in range(len(test1_labels)):
 temp=np.array([])
 for j in range(1000):
 temp=np.append(temp,0)
 index1 = np.where(test1_labels[i]==1)
 index2 = np.where(test2_labels[i]==1)
 index3 = np.where(test3_labels[i]==1)
 s1=int(index1[0])
 s2=int(index2[0])
 s3=int(index3[0])
 s4=str(s1)
 s5=str(s2)
 s6=str(s3)
 s7=s4+s5+s6
 s8=int(s7)
 temp[s8]=1
 new_test_labels.append(temp)
new_test_labels=np.array(new_test_labels)
# Parameters
learning_rate = 0.001
training_epochs = 100
batch_size = 1000
display_step = 1
# Network Parameters
n_hidden_1 = 392 # 1st layer number of neurons
n_hidden_2 = 196 # 2nd layer number of neurons
n_input = 2352 # MNIST data input
n_classes = 1000 # MNIST total classes
# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])
# Storing layers weight & bias
weights1 = tf.Variable(
tf.truncated_normal([n_input, n_hidden_1],
stddev=1.0 / math.sqrt(float(n_input))),
name='weights')
biases1 = tf.Variable(tf.zeros([n_hidden_1]),
name='biases')
weights2 = tf.Variable(
tf.truncated_normal([n_hidden_1, n_hidden_2],
stddev=1.0 / math.sqrt(float(n_hidden_1))),
name='weights')
biases2 = tf.Variable(tf.zeros([n_hidden_2]),
name='biases')
weights3 = tf.Variable(
tf.truncated_normal([n_hidden_2, n_classes],
stddev=1.0 / math.sqrt(float(n_hidden_2))),
name='weights')
biases3 = tf.Variable(tf.zeros([n_classes]),
name='biases')
# Creating model
def multilayer_perceptron(x):
 layer_1 = tf.nn.tanh(tf.matmul(x, weights1) + biases1)
 layer_2 = tf.nn.tanh(tf.matmul(layer_1, weights2) + biases2)
 out_layer = tf.matmul(layer_2, weights3) + biases3
 return out_layer
# Constructing model
logits = multilayer_perceptron(X)
# Regularization
regularizers = tf.nn.l2_loss(weights1) + tf.nn.l2_loss(weights2)+tf.nn.l2_loss(weights3)
# loss = tf.reduce_mean(loss + beta * regularizers)
# Defining loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)+0.001 *
regularizers)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
# Initializing the variables
init = tf.global_variables_initializer()
# Testing model
pred = tf.nn.softmax(logits) # Applying softmax
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
# Calculating accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
Epoches_graph=[]
TrainingAccuracy_graph=[]
TestingAccuracy_graph=[]
Loss_graph=[]
with tf.Session() as sess:
 sess.run(init)
 # Training cycle
 for epoch in range(training_epochs):
 avg_cost = 0.
 total_batch = int(40000/batch_size)
 # Loop over all batches
 k=0
 h=0
 while k<total_batch:
 batch_x=new_train_images[h:h+batch_size]
 batch_y=new_train_labels[h:h+batch_size]
 # Run optimization op (backprop) and cost op (to get loss value)
 _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
 Y: batch_y})
 # Computing average loss
 avg_cost += c / total_batch
 k=k+1
 h=h+batch_size
 # Displaying logs per epoch step
# print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
 Training_Acc=accuracy.eval({X:new_train_images, Y:new_train_labels})
 Testing_Acc=accuracy.eval({X:new_test_images, Y:new_test_labels})
# print('Training Accuracy: ',Training_Acc)
# print('Testing Accuracy: ',Testing_Acc)
 Epoches_graph.append(epoch)
 TrainingAccuracy_graph.append(Training_Acc)
 TestingAccuracy_graph.append(Testing_Acc)
 Loss_graph.append(avg_cost)
 print("Optimization Finished!")

print('Final Training Accuracy: ',Training_Acc)
print('Final Testing Accuracy: ',Testing_Acc)

import matplotlib.pyplot as plt
plt.xlabel('Number of Epoches')
plt.ylabel('Loss')
plt.scatter(Epoches_graph,Loss_graph,color='red')
plt.show()
import matplotlib.pyplot as plt1
plt1.xlabel('Number of Epoches')
plt1.ylabel('Training Accuracy')
plt1.scatter(Epoches_graph,TrainingAccuracy_graph,color='blue')
plt1.show()
import matplotlib.pyplot as plt2
plt2.xlabel('Number of Epoches')
plt2.ylabel('Testing Accuracy')
plt2.scatter(Epoches_graph,TestingAccuracy_graph,color='green')
plt2.show()
