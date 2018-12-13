# Import MNIST data
import math
import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("C:/Users/krta225/Downloads/mnistData", one_hot=True)
images,labels=mnist.train.next_batch(5000)
# Parameters
learning_rate = 0.001
training_epochs = 70
batch_size = 1000
display_step = 1
# Network Parameters
n_hidden_1 = 300 # 1st layer number of neurons
n_hidden_2 = 150 # 2nd layer number of neurons
n_input = 784 # MNIST data input
n_classes = 10 # MNIST total classes
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
 layer_1 = tf.nn.relu(tf.matmul(x, weights1) + biases1)
 layer_2 = tf.nn.relu(tf.matmul(layer_1, weights2) + biases2)
 out_layer = tf.matmul(layer_2, weights3) + biases3
 return out_layer
# Constructing model
logits = multilayer_perceptron(X)
# Regularization
regularizers = tf.nn.l2_loss(weights1) + tf.nn.l2_loss(weights2)+tf.nn.l2_loss(weights3)
# Defining loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)+0.001 *
regularizers)
optimizer = tf.train.AdamOptimizer(learning_ra
 sess.run(init)
 # Training cycle
 for epoch in range(training_epochs):
 avg_cost = 0.
 total_batch = int(5000/batch_size)
 # Loop over all batches
 k=0
 h=0
 while k<total_batch:
 batch_x=images[h:h+batch_size]
 batch_y=labels[h:h+batch_size]
 # Run optimization op (backprop) and cost op (to get loss value)
 _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
 Y: batch_y})
 # Computing average loss
 avg_cost += c / total_batch
 k=k+1
 h=h+batch_size
 # Displaying logs per epoch step
# print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
 Training_Acc=accuracy.eval({X:images, Y:labels})
 Testing_Acc=accuracy.eval({X:mnist.test.images, Y:mnist.test.labels})
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
