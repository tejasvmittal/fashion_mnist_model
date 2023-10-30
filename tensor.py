import os
import numpy as np
import tensorflow as tf
from tensorflow import keras


current_dir = os.getcwd()

data_path = os.path.join(current_dir, "data/mnist.npz")

# Get only training set
(training_images, training_labels), (test_images,
                                     test_labels) = tf.keras.datasets.mnist.load_data()



def reshape_and_normalize(images):

    # Reshape the images to add an extra dimension
    images = np.reshape(
        images, (images.shape[0], images.shape[1], images.shape[2], 1))

    # Normalize pixel values
    images = images/255
    return images


training_images = reshape_and_normalize(training_images)

print(f"Maximum pixel value after normalization: {np.max(training_images)}\n")
print(f"Shape of training set after reshaping: {training_images.shape}\n")
print(f"Shape of one image after reshaping: {training_images[0].shape}")


class myCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):
      if logs.get('accuracy') is not None and logs.get('accuracy') > 0.995:
          print("\nReached 99.5% accuracy so cancelling training!")
          self.model.stop_training = True



def convolutional_model():

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(
            32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        # 10 different classes of clothing
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


model = convolutional_model()

callbacks = myCallback()

history = model.fit(training_images, training_labels,
                    epochs=10, callbacks=[callbacks])
