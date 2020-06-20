# Earlier when you trained for extra epochs you had an issue where your loss might change.
# It might have taken a bit of time for you to wait for the training to do that,
# and you might have thought 'wouldn't it be nice if I could stop the training
# when I reach a desired value?' -- i.e. 95% accuracy might be enough for you,
# and if you reach that after 3 epochs, why sit around waiting for it to finish
# a lot more epochs....So how would you fix that? Like any other program...you have callbacks!
# Let's see them in action.


import tensorflow as tf
print(tf.__version__)
ACCURACY_THRESHOLD = 0.99

def train_mnist():
  # Please write your code only where you are indicated.
  # please do not remove # model fitting inline comments.

  # YOUR CODE SHOULD START HERE
  class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
      if (logs.get('acc') > ACCURACY_THRESHOLD):
        print("\nReached 99% accuracy so cancelling training!")
        self.model.stop_training = True

  callbacks = myCallback()
  # YOUR CODE SHOULD END HERE

  mnist = tf.keras.datasets.mnist

  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  # YOUR CODE SHOULD START HERE
  x_train = x_train / 255.0
  x_test = x_test / 255.0
  # YOUR CODE SHOULD END HERE
  model = tf.keras.models.Sequential([
    # YOUR CODE SHOULD START HERE
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    # YOUR CODE SHOULD END HERE
  ])

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  # model fitting
  history = model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])
  # model fitting
  return history.epoch, history.history['accuracy'][-1]