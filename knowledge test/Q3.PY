import tensorflow as tf
from tensorflow.keras.losses import Loss

# Define a custom loss function
class CustomLoss(Loss):
    def __init__(self, threshold=1.0, **kwargs):
        super(CustomLoss, self).__init__(**kwargs)
        self.threshold = threshold

    def call(self, y_true, y_pred):
        # Compute Mean Absolute Error
        mae = tf.reduce_mean(tf.abs(y_true - y_pred))
        # Apply a penalty for errors above the threshold
        penalty = tf.reduce_mean(tf.maximum(0.0, tf.abs(y_true - y_pred) - self.threshold))
        return mae + penalty

# Example usage with a Keras model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Create a simple model
model = Sequential([
    Dense(64, activation='relu', input_shape=(10,)),
    Dense(1)
])

# Compile the model with the custom loss function
model.compile(optimizer=Adam(), loss=CustomLoss(threshold=1.0))

# Dummy data
import numpy as np
x_train = np.random.random((100, 10))
y_train = np.random.random((100, 1))
x_test = np.random.random((20, 10))
y_test = np.random.random((20, 1))

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=10, validation_data=(x_test, y_test))
