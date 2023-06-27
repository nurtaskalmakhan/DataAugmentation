import matplotlib.pyplot as plt
import numpy as np
import cv2 
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

dataset_url = 
data_dir = tf.keras.utils.get_file()

import pathlib
data_dir = pathlib.Path(data_dir)

list(data_dir.glob(''))


PIL.Image.open(str(roses[1]))

X, y = [], []

for flower_name, images in flower_images.dict.items():
  print(flower_name)
  print(len(images))


for flower_name, images in flower_images.dict.items():
  for image in images:
    img = cv2.imread(str(image))
    resized_img = cv2.resize(img,(180,180))
    X.append(resized_img)
    y.append(flowers_labels_dict)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

model = Sequental([
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.evaluate(X_test_scaled, y _test)
predictions = model.predict(X_test_scaled)
score = tf.nn.softmax(predictions[0])
np.argmax(score)

data_augmentation = keras.Sequental([
    layers.experimental.preprocessing.RandomZoom(0,3),
])

plt.axis('off')
plt.imshow(X[0])

plt.axis('off')
plt.imshow(data_augmentation(X)[0].numpy().astype("uint8"))

num_classes = 5

model = Sequental([
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0,2),
    layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(X_train_scaled, y_train, epochs = 30)

model.evaluate(X_test_scaled, y_test)