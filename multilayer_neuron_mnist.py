import input_data
mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)

import numpy as np
import tensorflow as tf

batch_size = 100
learning_rate = 0.005
training_epochs = 20

 # 28*28 total pixels
n_in = 784
n_hidden = 256
n_out = 10

# Create graph
x = tf.placeholder("float", [None, n_in])
y = tf.placeholder("float", [None, n_out])

# Create weights: n_out weights for each pixel (n_in)
# 2 hidden layers
w0 = tf.Variable(tf.random_normal([n_in, n_hidden]))
w1 = tf.Variable(tf.random_normal([n_hidden, n_hidden]))
w = tf.Variable(tf.random_normal([n_hidden, n_out]))

b0 = tf.Variable(tf.random_normal([n_hidden]))
b1 = tf.Variable(tf.random_normal([n_hidden]))
b = tf.Variable(tf.random_normal([n_out]))

# model = tf.add(tf.matmul(x, w), b)
l1 = tf.nn.relu(tf.add(tf.matmul(x, w0), b0))
l2 = tf.nn.relu(tf.add(tf.matmul(l1, w1), b1))
model = tf.add(tf.matmul(l2, w), b)

# Softmax & cross entropy loss
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(model, y))
optimize = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.Session() as sess:
	tf.initialize_all_variables().run()

	for epoch in range(training_epochs):
		avg_cost = 0.0
		total_batch = int(mnist.train.num_examples / batch_size)
		for i in range(total_batch):
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			sess.run(optimize, feed_dict={x: batch_xs, y: batch_ys})
			avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys}) / total_batch
		print "Epoch: %04d" % (epoch+1), "cost=", "{:.6f}".format(avg_cost)

	# Test model
	correct = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct, "float"))
	print "Accuracy: ", accuracy.eval({x: mnist.test.images, y: mnist.test.labels})