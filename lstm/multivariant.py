import logging

import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from keras.models import Sequential

from plotlywrapper import basicLineChart

from utils import initLogging

def trainMultiInput(data, nSteps, epochs=200, verbose=0, tests=None):
    X, y = splitSequence(data, nSteps)

    logging.info("Shape input: %s", X.shape)
    logging.info("  Number of samples : %s", X.shape[0])
    logging.info("  Number of samples per step : %s", X.shape[1])
    logging.info("  Number of features : %s", X.shape[2])
    logging.info("Shape targer: %s", y.shape)
    
    model = Sequential()
    model.add(layers.LSTM(50, activation='relu', input_shape=(nSteps, X.shape[2])))
    model.add(layers.Dense(1))

    model.summary(print_fn=logging.warning)
    
    model.compile(optimizer='adam', loss='mse')

    model.fit(X, y, epochs=epochs, verbose=verbose)


    if tests is not None:
        for vals, expectation in tests:
            yhat = model.predict(vals, verbose=verbose)
            logging.debug("Predicting from: %s", vals)
            logging.info("Predicted %s | expected %s", yhat, expectation)
            
            
def trainParallel(data, nSteps, epochs=200, verbose=0, tests=None, LSTMType="Regular"):
    X, y = splitSequence_parallel(data, nSteps)
    logging.info("Shape input: %s", X.shape)
    logging.info("Shape targer: %s", y.shape)

    logging.debug("X : %s ", X)
    logging.debug("y : %s ", y)
    
    model = Sequential()
    if LSTMType == "Stacked":
        model.add(layers.LSTM(50, activation='relu', return_sequences=True, input_shape=(nSteps, X.shape[2])))
        model.add(layers.LSTM(50, activation='relu'))
        model.add(layers.Dense(X.shape[2]))
    elif LSTMType == "Regular":
        model.add(layers.LSTM(50, activation='relu', input_shape=(nSteps, X.shape[2])))
        model.add(layers.Dense(X.shape[2]))
    else:
        raise NotImplementedError
    model.summary(print_fn=logging.warning)
    
    model.compile(optimizer='adam', loss='mse')

    model.fit(X, y, epochs=epochs, verbose=verbose)


    if tests is not None:
        for vals, expectation in tests:
            yhat = model.predict(vals, verbose=verbose)
            logging.debug("Predicting from: %s", vals)
            logging.info("Predicted %s | expected %s", yhat, expectation)

    
def transformDataset(inputs, target):
    reshaped_inputs = []
    for input_ in inputs:
        reshaped_inputs.append(
            input_.reshape((len(input_), 1))
        )

    return np.hstack(reshaped_inputs + [target.reshape((len(input_), 1))])

def splitSequence(data, steps):
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + steps
        if end_ix > len(data)-1:
            break
        data_x, data_y = data[i:end_ix, :-1], data[end_ix-1, -1]
        X.append(data_x)
        y.append(data_y)

    return np.array(X), np.array(y)

def splitSequence_parallel(data, steps):
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + steps
        if end_ix > len(data)-1:
            break
        data_x, data_y = data[i:end_ix, :], data[end_ix, :]
        X.append(data_x)
        y.append(data_y)
    return np.array(X), np.array(y)


if __name__ == "__main__":
    initLogging(20)

    if True:
        length = 20
        
        inputSeq1 = np.array([x*10 for x in range(1,length+1)])
        inputSeq2 = np.array([5+(x*10) for x in range(1,length+1)])
        targetSeq = np.array([inputSeq1[i]+inputSeq2[i] for i in range(len(inputSeq1))])

        basicLineChart([pd.Series(inputSeq1), pd.Series(inputSeq2), pd.Series(targetSeq)],
                       ["Sequence 1","Sequence 2","Target"],
                       ["lines+markers","lines+markers","lines+markers"],
                       savePlot=True, filename="multivariant_sum")
        
        allInputs = [inputSeq1, inputSeq2]
        nInptus = len(allInputs)
        data = transformDataset(allInputs, targetSeq)



        if False:
            testSeries = [
                (np.array([[80, 85], [90, 95], [100, 105]]).reshape((1, 3, 2)) , 205),
                (np.array([[180, 185], [190, 195], [200, 205]]).reshape((1, 3, 2)) , 405),
            ]
            trainMultiInput(data, 3, tests=testSeries)

        if True:
            testSeries = [
                (np.array([[80, 85, 165], [90, 95,185], [100, 105, 205]]).reshape((1, 3, 3)), [110, 115, 225]),
                (np.array([[180, 185, 365], [190, 195,385], [200, 205, 405]]).reshape((1, 3, 3)), [210, 215, 425]),
            ]
            logging.warning("Regular LSTM")
            trainParallel(data, 3, tests=testSeries, LSTMType="Regular")
            logging.warning("Stacked LSTM")
            trainParallel(data, 3, tests=testSeries, LSTMType="Stacked")
