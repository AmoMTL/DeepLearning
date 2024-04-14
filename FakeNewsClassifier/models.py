import pandas as pd
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint

def CreateModelsLogDir(model_folder_name="models", log_folder_name="logs", logdir=True, modeldir=True):
        model_dir = f"{model_folder_name}"
        if modeldir:
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
        logs_dir = f"{log_folder_name}"
        if logdir:
            if not os.path.exists(logs_dir):
                os.makedirs(logs_dir)
        return model_dir, logs_dir

def CheckpointCallback(model_dir, freq="epoch", save_weights_only=False):
    checkpoint_path = model_dir + "/cp-{epoch:04d}.ckpt"
    callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=save_weights_only, save_freq=freq)
    return callback

def PreProcess(article):
        tokenizer = Tokenizer(num_words=5000)
        sequences = tokenizer.texts_to_sequences([article])
        padded_sequences = pad_sequences(sequences, maxlen=200)
        return padded_sequences

class LSTMBinaryClassifier():
    def __init__(self, model_name="LSTM_model"):
        self.model_name = model_name
        
    def Learn(self, df, epochs=10, val_split=0.2, batch_size=64, save_checkpoints=True):
        self.df = df
        self.model = self.CreateModel()
        X, y = self.SplitXy()
        X_padded = self.PreProcessFit(X)
        if save_checkpoints:
            self.model.fit(X_padded, y, epochs=epochs, validation_split=val_split, batch_size=batch_size, callbacks=[CheckpointCallback(self.model_dir)])
        else:
            self.model.fit(X_padded, y, epochs=epochs, validation_split=val_split, batch_size=batch_size)
        self.model_dir, _ = CreateModelsLogDir(logdir=False)
        model_path = os.path.join(self.model_dir, self.model_name)
        self.model.save(model_path)

    def TestOne(self, article, model_path):
        model = load_model(model_path)
        
        X = PreProcess(article)
        prediction = model.predict(X)
        predicted_label = (prediction > 0.5).astype(int)
        print(f"Model predicts {predicted_label}")

    def PreProcessFit(self, X):
        tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
        tokenizer.fit_on_texts(X)
        sequences = tokenizer.texts_to_sequences(X)
        padded_sequences = pad_sequences(sequences, maxlen=200, padding='pre', truncating='pre')
        return padded_sequences

    def SplitXy(self):
        X = self.df.iloc[:,0]
        y = self.df.iloc[:,1]
        return X, y

    def CreateModel(self, optimizer="adam", loss='binary_crossentropy', lstm_nodes=128, dense_nodes=128):
        model = Sequential()
        model.add(Embedding(input_dim=5000, output_dim=64, input_length=200))
        model.add(LSTM(lstm_nodes))
        model.add(Dropout(0.2))
        model.add(Dense(dense_nodes, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation="sigmoid"))
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        return model