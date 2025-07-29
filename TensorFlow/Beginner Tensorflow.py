# Source: https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/quickstart/beginner.ipynb#scrollTo=h3IKyzTCDNGo

# Setup
import tensorflow as tf
import numpy as np
print("-----TensorFlow version:", tf.__version__)


# Get Dataset 
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#Build machine learning model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

# predictions...
predictions = model(x_train[:1]).numpy()
print(predictions)

#...to probabilities
probs = tf.nn.softmax(predictions).numpy()
print(probs)

#loss function
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
print(loss_fn(y_train[:1], predictions).numpy())

# Model Compile
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# Fit Model
model.fit(x_train, y_train, epochs=5)

# Evaluate against y_train
print(model.evaluate(x_test,  y_test, verbose=2))

# Create probability model
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

print(probability_model(x_test[:5])) 
