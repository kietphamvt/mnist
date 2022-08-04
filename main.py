import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Sequential
from matplotlib import pyplot as plt
print("TensorFlow version:", tf.__version__)
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# x_train = np.expand_dims(x_train, axis=-1)
# x_train = tf.image.resize(x_train, [40, 40]) # if we want to resize

# print(x_train.shape)
# image_temp = x_train[:1][0].numpy()
# for i, v1 in enumerate(image_temp):
#     for j, v2 in enumerate(v1):
#         if v2 != 255 and v2 != 0:
#             image_temp[i, j] = 255.0

# # image_temp /= 255.0
# image_temp = np.divide(image_temp, 255.0)
# plt.imshow(image_temp, cmap='gray')
# plt.show()
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()
tf.nn.softmax(predictions).numpy()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(y_train[:1], predictions).numpy()
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)

probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])
plt.imshow(x_test[:1][0])
plt.show()
print(np.argmax(probability_model(x_test[:1])))
