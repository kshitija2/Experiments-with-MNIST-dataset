# Import MNIST data
import math
import numpy as np
import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("C:/Users/krta225/Downloads/mnistData", one_hot=True)
# Following code is to create total 55 (0 to 54) labels for unordered images
a=[]
for i in range(100):
 i=str(i)
 if len(i)==1:
 i='0'+i
 a.append(i)

b={}
i=0
for each in a:
 if int(each) not in b.keys():
 b.update({int(each):i})
 s=each[1]+each[0]
 if int(s) not in b.keys():
 b.update({int(s):i})
 i=i+1
train_images=np.concatenate((mnist.train.images,mnist.validation.images))
train_labels=np.concatenate((mnist.train.labels,mnist.validation.labels))
test_images=mnist.test.images
test_labels=mnist.test.labels
#Taking first 50000 and last 50000 training images in 2 rows
images1=train_images[0:50000]
images2=train_images[10000:60000]
new_train_images=[]
# Adding pixels of two images in one column
for i in range(len(images1)):
 temp1=images1[i]*255
 temp2=images2[i]*255
 temp3=images1[i]+images2[i]
 temp=temp3/255
 new_train_images.append(temp)
new_train_images=np.array(new_train_images)
# Creating labels for images
labels1=train_labels[0:50000]
labels2=train_labels[10000:60000]
new_train_labels=[]
for i in range(len(labels1)):
 temp=np.array([])
 for j in range(55):
 temp=np.append(temp,0)
 index1 = np.where(labels1[i]==1)
 index2 = np.where(labels2[i]==1)
 s1=int(index1[0])
 s2=int(index2[0])
 s4=str(s1)
 s5=str(s2)
 s7=s4+s5
 s8=int(s7)
 s9=b.get(s8)
 temp[s9]=1
 new_train_labels.append(temp)
new_train_labels=np.array(new_train_labels)
test1=test_images[0:9000]
test2=test_images[1000:10000]
new_test_images=[]
for i in range(len(test1)):
 temp1=test1[i]*255
 temp2=test2[i]*255
 temp3=test1[i]+test2[i]
 temp=temp3/255
 new_test_images.append(temp)
new_test_images=np.array(new_test_images)
test1_labels=test_labels[0:9000]
test2_labels=test_labels[1000:10000]
new_test_labels=[]
for i in range(len(test1_labels)):
 temp=np.array([])
 for j in range(55):
 temp=np.append(temp,0)
 index1 = np.where(test1_labels[i]==1)
 index2 = np.where(test2_labels[i]==1)
 s1=int(index1[0])
 s2=int(index2[0])
 s4=str(s1)
 s5=str(s2)
 s7=s4+s5
 s8=int(s7)
 s9=b.get(s8)
 temp[s9]=1
 new_test_labels.append(temp)
new_test_labels=np.array(new_test_labels)
# Parameters
learning_rate = 0.001
training_epochs = 100
batch_size = 100
display_step = 1
# Network Parameters
n_hidden_1 = 300 # 1st layer number of neurons
n_hidden_2 = 200 # 2nd layer number of neurons
n_input = 784 # MNIST data input
n_classes = 55 # MNIST total classes
# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])
# Storing layers weight & bias
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
 layer_1 = tf.add(tf.matmul(x, weights1), biases1)
 layer_1=tf.nn.relu(layer_1)
 layer_2 = tf.add(tf.matmul(layer_1, weights2), biases2)
 layer_2=tf.nn.relu(layer_2)
 out_layer = tf.matmul(layer_2, weights3) + biases3
 return out_layer
# Regularization
#reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
#reg_constant = 0.1
# Constructing model
logits = multilayer_perceptron(X)
# Defining loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
 logits=logits, labels=Y))
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
 total_batch = int(50000/batch_size)
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
