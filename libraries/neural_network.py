"""
A file used to build, train, and test the neural network.
"""
import math
import pandas as pd
import numpy as np
from scipy import stats
from keras import backend as K
import tensorflow as tf

from keras import models
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from numpy.typing import ArrayLike

def build_loss(W):
    
    def loss_function(
        y_true: ArrayLike, 
        y_pred: ArrayLike, 
        custom_loss: bool = True
    ) -> ArrayLike:
        """

        Parameters
        ----------

        Returns
        -------
        """
        R = y_true - y_pred

        if custom_loss == True:
            R_trans = tf.transpose(R)
            W_corr = tf.slice(W, [0,0],[K.shape(y_true)[0], K.shape(y_true)[0]])
            loss = tf.squeeze(tf.matmul(R_trans, tf.matmul(W_corr, R)), axis=-1)

        else:
            loss = K.mean(K.square(R), axis=-1)
            
        return loss
        
    return loss_function

def correlation_matrix(
    custom_loss: bool, 
    N: int
) -> ArrayLike:
    """

    Parameters
    ----------

    Returns
    -------
    """
    if custom_loss == True:
        W = K.constant(np.tri(N, N, 1) - np.tri(N, N, -2), dtype=tf.float32)
    else:
        W = K.constant(np.identity(N), dtype=tf.float32)
    
    return W


class NeuralNetwork():
    """
    A class used to build a Neural Network.
    
    Parameters
    ----------
    """
    def __init__(
        self, 
        model, 
        data_shape, 
        scaler, 
        custom_loss, 
        l2_reg, 
        l2_lambda=5e-4
    ):
        self.model = model
        self.scaler = scaler
        self.custom_loss = custom_loss
        self.l2_reg = l2_reg
        self.l2_lambda = l2_lambda
        
        self.matrix_shape = data_shape[0]
        self.total_inputs = data_shape[1]
        
        self.epoch = 5000
        self.verbose = 0
        self.batch_train = self.batch_test = 32
        
    def initialize(self):
        """

        Parameters
        ----------

        Returns
        -------
        """
        network = self.build_neural_network()
        W = correlation_matrix(self.custom_loss, self.matrix_shape) # N=1000
        network.compile(optimizer="Adam", loss=build_loss(W), 
                        metrics=["mse", build_loss(W)])
        network.save_weights(self.model)

        print("Model successfully built!")

        return network
    
    def build_neural_network(self):
        """

        Parameters
        ----------

        Returns
        -------
        """
        l2_reg = self.l2_reg
        l2_lambda = self.l2_lambda
        
        network = models.Sequential()

        if l2_reg:
            regularization = regularizers.l2(l2_lambda)
        else:
            regularization = None

        network.add(layers.Dense(units=8, input_dim=self.total_inputs,
                                 activation="sigmoid",
                                 kernel_initializer="glorot_uniform",
                                 bias_initializer="glorot_uniform",
                                 kernel_regularizer=regularization))
        network.add(layers.Dense(units=1, activation="linear",
                                 kernel_initializer="glorot_uniform",
                                 bias_initializer="glorot_uniform",
                                 kernel_regularizer=regularization))
        return network
    
    def train(
        self, 
        X: ArrayLike, 
        y: ArrayLike, 
        network, 
        num_train: int, 
        best_model: str, 
        mean: float, 
        std_dev: float, 
        patience: int, 
        test_size: float = 0.15, 
        initial_train: bool = False
    ) -> None:
        """

        Parameters
        ----------

        Returns
        -------
        """
        model, scaler = self.model, self.scaler
        verbose, epoch = self.verbose, self.epoch
        batch_train, batch_test = self.batch_train, self.batch_train
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)

        X_train = scaler.fit_transform(X_train)
        X_train = self.gaussian_weights(X, X_train, mean, std_dev)
        X_test = scaler.transform(X_test)
        X_data = [X_train, X_test]
        y_data = [y_train, y_test]
        
        callbacks = self.get_callbacks(patience, best_model)
        network = self.load_network_weights(network, best_model, initial_train)

        network = self.fit_NN(network, X_data, y_data, callbacks, epoch, 
                              batch_train, num_train, verbose, model, best_model)
        y_pred = network.predict(X_test, batch_size=batch_test)
        score = network.evaluate(X_test, y_test, batch_size=batch_test, verbose=verbose)

#         results = pd.DataFrame(scaler.inverse_transform(X_test), \
#                                index=range(y_test.shape[0]), columns=self.cols)
#         results["E_test"], results["E_pred"] = y_test, y_pred    
#         results = results.sort_values(["Nmax", "hw", "Ediff"]).reset_index(drop=True)

        return None

    def compute_gauss_weights(
        self, 
        X: ArrayLike, 
        x1: float, 
        x2: float, 
        mean: float, 
        std_dev: float, 
        shift: bool = True
    ) -> ArrayLike:
        """

        Parameters
        ----------

        Returns
        -------
        """
        if shift:
            mean = (mean - x1) / (x2 - x1)
            std_dev = abs((std_dev - x1) / (x2 - x1))
        return np.exp(-(X[:, 0] - mean)**2 / std_dev)

    def gaussian_weights(
        self, 
        X: ArrayLike, 
        X_train: ArrayLike, 
        mean: float, 
        std_dev: float
    ) -> ArrayLike:
        """
        """
        x1, x2 = X["hw"].iloc[0], X["hw"].iloc[-1]
        X_gauss = self.compute_gauss_weights(X_train, x1, x2, mean, std_dev)
        
        if self.total_inputs == 3:
            X_set = np.vstack([X_gauss, X_train[:, 1], X_train[:, 2]]).T
        else:
            X_set = np.vstack([X_gauss, X_train[:, 1]]).T
        
        return X_set

    def get_callbacks(
        self, 
        patience: int, 
        best_model: str
    ) -> list:
        """

        Parameters
        ----------

        Returns
        -------
        """
        callbacks = [EarlyStopping(monitor="val_loss",\
                                   patience=patience,\
                                   restore_best_weights=True),
                     ModelCheckpoint(filepath=best_model, \
                                     monitor="val_loss", 
                                     save_best_only=True)]

        return callbacks

    def load_network_weights(
        self, 
        network, 
        best_model: str, 
        initial_train: bool
    ):
        """

        Parameters
        ----------

        Returns
        -------
        """
        if initial_train:
            network.load_weights(self.model)
        else: 
            network.load_weights(best_model)

        return network

    def fit_NN(
        self,
        network,
        X: ArrayLike, 
        y: ArrayLike, 
        callbacks: list,
        epoch: int,  
        batch_train: int, 
        num_train: int,  
        verbose: int,
        model: str, 
        best_model: str
    ):
        """

        Parameters
        ----------

        Returns
        -------
        """
        for m in range(num_train):
            history = network.fit(X[0], y[0], epochs=epoch,\
                                  batch_size=batch_train, verbose=verbose,\
                                  callbacks=callbacks, validation_data=(X[1], y[1]))

            if verbose != 0:
                print("\n")

            network.save(best_model)
            network.load_weights(best_model)

        return network
    
    def predict(
        self, 
        hw: int, 
        Nmax: int,
        model
    ) -> ArrayLike:
        """

        Parameters
        ----------

        Returns
        -------
        """
        if self.total_inputs == 3:
            pred_input = [hw, Nmax, 0]
            cols = ["hw", "Nmax", "Ediff"]
        else:
            pred_input = [hw, Nmax]
            cols = ["hw", "Nmax"]
        
        scaler = self.scaler
        df = pd.DataFrame([pred_input], columns=cols)        
        X = scaler.transform(df[cols])

        return pd.Series(model.predict(X).flatten())




