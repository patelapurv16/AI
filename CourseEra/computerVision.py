import tensorflow as tf
print(tf.__version__)
import matplotlib.pyplot as plt
print("Imported Matplotlib PyPlot")
import numpy as np
np.set_printoptions(linewidth=200)

mnist = tf.keras.datasets.fashion_mnist

print("Data Loading....")
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
print("Successful - Data Loading")

plt.imshow(training_images[34])
plt.show()
print(training_labels[0])
print(training_labels[0])

#Neural network works best with normalizing the data
training_images = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10,activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10,activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

#Sequence: That defines the Sequence of layers in the neural network
# Flatten: Flatten takes the shape and turns it into a 1 dimensional set
# Dense: Adds a layer of neurons
# Each layers of neurons need an activation function to tell them what to do. There's lot of options, but just use these for now
# Relu effectively means "If X>0 return X, else return 0" - So it only passes values 0 or greater to the nest layer in the network
# SOftmax takes a set of values, effectively picks the biggest one, so for example, if the output of the last layer looks like [0.1,0.1,0.05, 0.1, 9.5, 0.1, 0.05, 0.05, 0.05]
# it saves you from fishing through it looking for the biggest value, and turns it into [0,0,0,0,1,0,0,0,0] - the goal is to save a lot of coding]

model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs = 10)
print(model.evaluate(test_images, test_labels))

#Classification of the test images
classifications=model.predict(test_images)
print(classifications[0])
print(test_labels[0])



