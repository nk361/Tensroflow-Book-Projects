import tensorflow as tf
# from tensorflow import keras as ks

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()  # mnist is a dataset of labeled, hand drawn digits for supervised learning, 60k for training, 10k for testing
x_train, x_test = x_train / 255.0, x_test / 255.0  # x trains are 60k images, test is 10k new images of hand drawn digits
#removing this line greatly decreases accuracy and increases trainging time
#you divide by 255 because the grayscale is 0-255 and so it makes it 0-1 instead

model = tf.keras.models.Sequential([  # I didn't really know that you could build your model all inline within sequential()
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # flattens each image into a 1D array for input, 28 by 28 is the image's sizes, no third dimension for color because they're in black and white
    #tf.keras.layers.Dense(512, activation=tf.nn.leaky_relu)  # normal relu neurons can die by going passed zero where the gradient is 0, leaky relu gives a small amount of gradient passed zero
    tf.keras.layers.Dense(512, activation=tf.nn.relu),  # like exponential function but compressed, better and newer than sigmoid I think and also maybe better than tanh too
    tf.keras.layers.Dropout(0.2),  # dropout randomly drops nodes with the parameter being the fraction rate to drop them
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)  # softmax is a great activation function for the final layer of a classification network, it gives values that can be converted to percents that WILL add up to be 100%
])
model.compile(optimizer='adam',  # Adam is one of many functions for traversing to a low point on a gradient, currently the best mini batch SGD optimizer, but may take more computations
              loss='sparse_categorical_crossentropy',  # categorical crossentropy is also called softmax loss, it is softmax activation plus crossentropy, can train a CNN to output probability over C classes, sparse in front just means it deals with integer input when you need "one-hot" encoded data which is like a tensor where each column has one true value in the whole row per class as a unique identifier of that classification, multiple being true in one column means multiple classes found
              metrics=['accuracy'])  # This is just for output, not for any change to the model. You can choose many others just to see how your model is doing

model.fit(x_train, y_train, epochs=5)  # this runs the samples through the model this many times, you should consider size of sample data to determine if it needs run through multiple times
model.evaluate(x_test, y_test)  # runs the test images through the trained model in batches

# ETA is learning rate, how much it changes weights depending on loss function

# You can add parameters to model.evaluate for batch size and more
