import numpy as np
import os
import time
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dropout, Conv2D, MaxPooling2D, Flatten, Dense, Activation
from tensorflow.keras.callbacks import TensorBoard

img_size = 50
val_split = 0.15

model_name = f'cats-dogs-64x2-{int(time.time())}'

tensorboard = TensorBoard(log_dir=r'logs/{}'.format(model_name))




# Loading the cats and dogs dataset

X = pickle.load(open("cats_dogs_X.pickle","rb"))
y = pickle.load(open("cats_dogs_y.pickle","rb"))

# Constructing the model

model = tf.keras.models.Sequential()

# 1st Conv2D layer with 64 nodes
model.add(tf.keras.layers.Conv2D(64,(3,3),input_shape=X.shape[1:]))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

# 2nd Conv2D layer with 64 nodes
model.add(tf.keras.layers.Conv2D(64,(3,3)))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

# Convert to 1D feature vectors
model.add(tf.keras.layers.Flatten())

# 3rd Dense layer
model.add(tf.keras.layers.Dense(64))

model.add(tf.keras.layers.Dense(1))
model.add(Activation('sigmoid'))

loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Compile the model
model.compile(optimizer='adam',
             loss=loss_fn,
             metrics=['accuracy'])

# Normalize the data and convert to numpy arrays
X = np.array(X).reshape(-1,img_size,img_size,1)
y = np.array(y)
X = tf.keras.utils.normalize(X,axis=1)

model.fit(X,y,epochs=10,validation_split=val_split, batch_size=32,callbacks=[tensorboard])
