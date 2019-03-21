# XOR implementation in Tensorflow with hidden layers being sigmoid to
# introduce non-linearity

import tensorflow as tf

# Create placeholders for training input and output labels

# my own comments
# placeholder throws an error if evaluated, needs values fed in from feed_dict, the optional argument to Session.run()
# placeholders are not compatible with eager execution
x_ = tf.placeholder(tf.float32, shape=[4, 2], name="x-input")
y_ = tf.placeholder(tf.float32, shape=[4, 1], name="y-input")

# Define the weights to the hidden and output layers respectively

# my own comments
# if trainable is true, it is added to the graph collection that will changed with the optimizer
# random_uniform takes a shape and minval and maxval, minval included, maxval excluded
w1 = tf.Variable(tf.random_uniform([2, 2], -1, 1), name="Weights1")
w2 = tf.Variable(tf.random_uniform([2, 1], -1, 1), name="Weights2")

# Define the bias to the hidden and output layers respectively

# my own comments
# zeros makes a tensor of that shape full of zeros as values
# you can also use zeros_like on an existing tensor to get zeros of the same shape and type as the existing tensor
# you can call trainable_variables to return all variables created with trainable = true
b1 = tf.Variable(tf.zeros([2]), name="Bias1")
b2 = tf.Variable(tf.zeros([1]), name="Bias2")

# Define the final output through forward pass

# my own comment
# for some reason, the shapes of the tensors aren't always compatible with matmul's needs and so the program fails
# z2 is the "hidden layer" and without sigmoid in its computation, it is linear and gives very poor results with all output almost equal
# when the hidden layers are linear, the whole model remains linear which makes it very poor with non linear problems like discriminating classes
z2 = tf.sigmoid(tf.matmul(x_, w1) + b1)
pred = tf.sigmoid(tf.matmul(z2, w2) + b2)

# Define the Cross-entropy/Log-loss Cost function based on the output label y and
# the predicted probability by the forward pass

# my own comments
# there is also reduce sum, reduce min, reduce max, etc for operations on a tensor
# log computes the natural log
# compute gradients is the first part of minimize
# it computes the gradients of loss for the variables in var_list
# apply_gradients is the second part of minimize, it returns the operation that applies the gradient
# you can give a second parameter that will increment by one after updating your variables
# get_slot lets you access additional variables your optimizer may use, like momentum or vars for accumulation means
cost = tf.reduce_mean(((y_ * tf.log(pred)) + ((1 - y_) * tf.log(1.0 - pred))) * -1)
learning_rate = 0.01
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Now that we have all that set up, we will start training

XOR_X = [[0, 0], [0, 1], [1, 0], [1, 1]]
XOR_Y = [[0], [1], [1], [0]]

# Initialize the variables

init = tf.initialize_all_variables()
sess = tf.Session()
writer = tf.summary.FileWriter("./Downloads/XOR_logs", sess.graph)

sess.run(init)
for i in range(100000):
    sess.run(train_step, feed_dict={x_: XOR_X, y_: XOR_Y})

print('Final Prediction', sess.run(pred, feed_dict={x_: XOR_X, y_: XOR_Y}))

sess.close()  # I added this

# my own comments
# possible output:

# Final Prediction [[0.01500467]
#  [0.98650396]
#  [0.9798039 ]
#  [0.0126513 ]]

# As you can see, it gives very close to true results when given only one true value
# and a very close value to false when given two trues or zero trues

# to see the tensorboard graph of this, run: tensorboard --logdir=./Downloads/XOR_logs
# in your virtual environment terminal
# if it doesn't work, run: tensorboard --inspect --logdir=./Downloads/XOR_logs
# to see if it can detect a .tfevents. file at that path
