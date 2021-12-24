import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import os
import librosa

from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

import warnings
warnings.filterwarnings('ignore')
import pdb

isVisualize = True
isMakeData = True
isLoadModel = False
isTrainModel = True

dataPath = "data"
visualPath = "visualization"

train = pd.read_csv(f"{dataPath}/train_master.tsv", "\t")
label = pd.read_csv(f"{dataPath}/label_master.tsv", "\t")
submit = pd.read_csv(f"{dataPath}/sample_submit.tsv", "\t", header=None)
#----------------------------

#----------------------------
if isMakeData:

    train167 = train.iloc[:167]
    train334 = train.iloc[167:334]
    train500 = train.iloc[334:]

    test167 = submit.iloc[:167]
    test334 = submit.iloc[167:334]
    test500 = submit.iloc[334:]

    def makeData(num, dataframe, filename):
        genre_x = np.zeros((0, 40))
        if filename == "train":
            file_name = "file_name"
        else:
            file_name = 0
        for file in dataframe[file_name]:
            file_path = f"{dataPath}/{filename}_sound_{num}/{file}"
            x1, x2 = librosa.load(file_path)
            mfcc = librosa.feature.mfcc(y=x1, sr=x2, n_mfcc=40)
            mean = np.mean(mfcc, axis=1)
            genre_x = np.vstack((genre_x, mean))
        df = pd.DataFrame(genre_x)
        if filename == "train":
            df["target"] = dataframe["label_id"].reset_index(drop=True)
        return df

    df_train = pd.concat([makeData(1, train167, "train"), makeData(2, train334, "train"), makeData(3, train500, "train")]).reset_index(drop=True)
    df_test = pd.concat([makeData(1, test167, "test"), makeData(2, test334, "test"), makeData(3, test500, "test")]).reset_index(drop=True)

    # save to pkl
    with open(f"{dataPath}/master_data.pkl","wb") as fp:
        pkl.dump(df_train,fp)
        pkl.dump(df_test,fp)

else:
    # load from pkl
    with open(f"{dataPath}/master_data.pkl","rb") as fp:
        df_train = pkl.load(fp)
        df_test = pkl.load(fp)
#----------------------------

#----------------------------
x = df_train.drop(["target"], axis=1)
y = df_train["target"]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

#----------------------------
class convMusic(keras.Model):
    #-------
    def __init__(self, visualPath="visualization"):
        super(convMusic, self).__init__()

        self.visualPath = visualPath

        # initialize
        self.dense1 = layers.Dense(256, activation="relu", input_shape=(40,))
        self.drop1 = layers.Dropout(0.25)
        self.dense2 = layers.Dense(128, activation="relu")
        self.drop2 = layers.Dropout(0.25)
        self.dense3 = layers.Dense(64, activation="relu")
        self.drop3 = layers.Dropout(0.25)
        self.dense4 = layers.Dense(32, activation="relu")
        self.drop4 = layers.Dropout(0.25)
        self.dense5 = layers.Dense(10, activation="sigmoid")
    #-------

    #-------
    def call(self, x):
        
        _ = self.dense1(x)
        _ = self.drop1(_)
        _ = self.dense2(_)
        _ = self.drop2(_)
        _ = self.dense3(_)
        _ = self.drop3(_)
        _ = self.dense4(_)
        _ = self.drop4(_)
        pred = self.dense5(_)

        return pred
    #-------

    #-------
    # train step
    def train_step(self,data):
        x, y_true = data

        with tf.GradientTape() as tape1:
            # predict
            y_pred = self(x, training=True)

            # train using gradients
            trainable_vars = self.trainable_variables

            loss = self.compiled_loss(y_true, y_pred, regularization_losses=self.losses)

        gradients = tape1.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # update metrics
        self.compiled_metrics.update_state(y_true, y_pred)

        # return metrics as dictionary
        return {m.name: m.result() for m in self.metrics}
    #-------

    #-------
    # test step
    def test_step(self, data):
        x, y_true = data
        
        y_pred = self(x, training=False)

        # loss
        self.compiled_loss(y_true, y_pred, regularization_losses=self.losses)

        # update metrics
        self.compiled_metrics.update_state(y_true, y_pred)

        # return metrics as dictionary
        return {m.name: m.result() for m in self.metrics}
    #-------

    #-------
    # predict step
    def predict_step(self, x):
        # predict
        y_pred = self(x, training=False)

        return y_pred
    #-------

#----------------------------

myModel = convMusic(visualPath=visualPath)

# make checkpoint callback to save trained parameters
checkpoint_path = "checkpoint/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)

if isLoadModel:
    # load trained parameters
    myModel.load_weights(checkpoint_path)

myModel.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["acc"])

if isTrainModel:
    # training
    hist = myModel.fit(x_train, y_train, epochs=500, batch_size=128, verbose=1, validation_data=(x_test, y_test), callbacks=[cp_callback])

if isVisualize:
    score = myModel.evaluate(x_test, y_test, verbose=1)
    print("accuracy=", score[1], "loss=", score[0])
    
    # accuracy
    plt.plot(hist.history["acc"])
    plt.plot(hist.history["val_acc"])
    plt.title("Accuracy")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()
    plt.savefig(f"{visualPath}/accuracy.pdf")

    # loss
    plt.plot(hist.history["loss"])
    plt.plot(hist.history["val_loss"])
    plt.title("Loss")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()
    plt.savefig(f"{visualPath}/loss.pdf")

pred = np.argmax(myModel.predict(df_test), axis=1)
submit[1] = pred
submit.to_csv("submission.tsv", index=False, header=False, sep="\t")

pdb.set_trace()
