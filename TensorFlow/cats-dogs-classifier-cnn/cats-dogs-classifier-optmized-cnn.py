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

# Loading the cats and dogs dataset

X = pickle.load(open("cats_dogs_X.pickle","rb"))
y = pickle.load(open("cats_dogs_y.pickle","rb"))

X = np.array(X).reshape(-1,img_size,img_size,1)
y = np.array(y)
X = tf.keras.utils.normalize(X,axis=1)


dense_layers = [0, 1, 2]
nodes = [32, 64, 128]
conv_layers = [1, 2, 3]

loss_fn = tf.keras.losses.BinaryCrossentropy()

for dense_layer in dense_layers:
    for node in nodes:
        for conv_layer in conv_layers:
            model_name = f"{conv_layer}x{node}-conv-{dense_layer}-dense-{int(time.time())}"
            tensorboard = TensorBoard(log_dir=r'logs_optimize2/{}'.format(model_name))
            print(model_name)
            model = tf.keras.models.Sequential()
            for i in range(conv_layer):
                if i==1:
                    model.add(tf.keras.layers.Conv2D(node,(3,3),input_shape=X.shape[1:]))
                    model.add(tf.keras.layers.Activation('relu'))
                    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
                else:
                    model.add(tf.keras.layers.Conv2D(node,(3,3)))
                    model.add(tf.keras.layers.Activation('relu'))
                    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
            model.add(tf.keras.layers.Flatten())
            for i in range(dense_layer):
                model.add(tf.keras.layers.Dense(node))
                model.add(tf.keras.layers.Activation('relu'))
            model.add(tf.keras.layers.Dense(1))
            model.add(tf.keras.layers.Activation('sigmoid'))
            model.compile(optimizer='adam',
                         loss=loss_fn,
                         metrics=['accuracy'])
            model.fit(X,y,epochs=10,callbacks=[tensorboard],validation_split=val_split,batch_size=32)
            model.save(r'C:\Users\Amo\Desktop\Amogha\Python\DeepLearning\Tensor Flow\cats-dogs-classifier-cnn\Models\{}'.format(model_name))






































