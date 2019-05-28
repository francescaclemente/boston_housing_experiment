import silence_tensorflow
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import extra_keras_metrics
from extra_keras_utils import set_seed
from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense, InputLayer, Dropout
#from plot_keras_history import plot_history
import pandas as pd

def mlp(epochs:int):
    set_seed(42)
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
    model = Sequential([
        InputLayer(input_shape=(x_train.shape[1],)),
        Dense(10, activation="relu"),
        Dense(10, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="nadam",
        loss="mse"
    )

    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=100,
        shuffle=True
    ).history

    model.save("model.h5")
    pd.DataFrame(history).to_csv("history.csv")
    #plot_history(history)
    plt.savefig("history.png")

    