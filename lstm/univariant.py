"""
Some tests for univariant time series forcasting. These are problems, where the predictor learns from a single series and learns to 
predict future behavior of it

Loosely following https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
"""
import logging

import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from keras.models import Sequential

from utils import initLogging

def generateInOutPatternSeries(sequence, steps, debug=False):
    """
    Rearrange a series into in- / output patterns with input sequence of len steps
    Example: 
      sequence = [1, 2, 3, 4, 5]; steps = 2
      --> input : [[1 , 2],[2 , 3], [3 , 4]]
      --> output : [[3], [4], [5]]

    Args:
      sequence (list, pd.Series, np.array)
      steps (int) : number of elements in input 

    Returns:
      retX (np.array) : 2D - Shape n*steps
      retY (np.array) : 1D - Shape n
    """
    sequence = list(sequence)
    
    logging.info("Got sequence: %s", sequence)

    X, y = [], []

    for i in range(len(sequence)):
        end_ix = i + steps
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
        if debug:
            logging.debug("Step %s | Input sequence : %s", i, seq_x)
            logging.debug("Step %s | Output sequence : %s", i, seq_y)

    retX, retY = np.array(X), np.array(y)
    if debug:
        logging.debug("Full Input sequence: %s", retX)
        logging.debug("Full Output sequence: %s", retY)
    
    return retX, retY

def trainRegularLSTM(sequence, nsteps, nfeatures, epochs, testSequence):
    """
    Train a LSTM with single LSTM layer follow by an output layer
    """

    X, y = generateInOutPatternSeries(sequence, 4)

    X = X.reshape((X.shape[0], X.shape[1], nfeatures))
    
    model = Sequential()
    # Set import shape according to timeseries.
    #  For univariant LSTM we only use one series so there is only "one feature"
    #  The nsteps is the number of steps are use to predict the next.
    model.add(layers.LSTM(50, activation='relu', input_shape=(nsteps, nfeatures)))
    model.add(layers.Dense(1))
    model.summary(print_fn=logging.warning)
    
    model.compile(optimizer='adam', loss='mse')

    logging.info("Starting to train %s epochs", epochs)
    model.fit(X, y, epochs=epochs, verbose=0)


    for testSeq, expectation in testSequence:
        x_input = np.array(testSeq)
        x_input = x_input.reshape((1, nsteps, nfeatures))
        yhat = model.predict(x_input, verbose=0)
        logging.info("Predicted %s as next step for %s", yhat, testSeq)
        logging.info("Expected %s", expectation)

def trainStackedLSTM(sequence, nsteps, nfeatures, epochs, stacks, testSequence):
    """
    Train a LSTM with stacked LSTM layer follow by an output layer
    """
    if stacks < 2:
        raise RuntimeError("At least 2 stacks are necessary")
    
    X, y = generateInOutPatternSeries(sequence, 4)

    X = X.reshape((X.shape[0], X.shape[1], nfeatures))
    
    model = Sequential()
    # Set import shape according to timeseries.
    #  For univariant LSTM we only use one series so there is only "one feature"
    #  The nsteps is the number of steps are use to predict the next.
    model.add(layers.LSTM(50, activation='relu', return_sequences=True, input_shape=(nsteps, nfeatures)))
    for i in range(1,stacks-1):
        logging.info("Adding LSTM layer w/ return_sequences=True")
        model.add(layers.LSTM(50, activation='relu', return_sequences=True))
    logging.info("Adding LSTM layer w/o return_sequences=True")
    model.add(layers.LSTM(50, activation='relu'))

    model.add(layers.Dense(1))
    model.summary(print_fn=logging.warning)

    model.compile(optimizer='adam', loss='mse')

    logging.info("Starting to train %s epochs", epochs)
    model.fit(X, y, epochs=epochs, verbose=0)


    for testSeq, expectation in testSequence:
        x_input = np.array(testSeq)
        x_input = x_input.reshape((1, nsteps, nfeatures))
        yhat = model.predict(x_input, verbose=0)
        logging.info("Predicted %s as next step for %s", yhat, testSeq)
        logging.info("Expected %s", expectation)

def trainBidirectionalLSTM(sequence, nsteps, nfeatures, epochs, testSequence):
    """
    Train a bidirectional LSTM with single LSTM layer follow by an output layer
    """

    X, y = generateInOutPatternSeries(sequence, 4)

    X = X.reshape((X.shape[0], X.shape[1], nfeatures))
    

    model = Sequential()
    # Note: input_shape is part of layer.Bidirectional not layer.LSTM!!!
    model.add(layers.Bidirectional(layers.LSTM(50, activation='relu'), input_shape=(nsteps, nfeatures)))
    model.add(layers.Dense(1))
    model.summary(print_fn=logging.warning)
    
    model.compile(optimizer='adam', loss='mse')

    logging.info("Starting to train %s epochs", epochs)
    model.fit(X, y, epochs=epochs, verbose=0)


    for testSeq, expectation in testSequence:
        x_input = np.array(testSeq)
        x_input = x_input.reshape((1, nsteps, nfeatures))
        yhat = model.predict(x_input, verbose=0)
        logging.info("Predicted %s as next step for %s", yhat, testSeq)
        logging.info("Expected %s", expectation)


if __name__ == "__main__":
    initLogging(10)


    if True:
        logging.info("Prediction next integer")
        seriesLength = 4    
        lengths = 200
        startVal = 1
        seriesLin = [1*x for x in range(1, lengths+1)]

        testSeries = [ ([(lengths+2)+x for x in range(seriesLength)], lengths+2+seriesLength),
                       ([(lengths+20)+x for x in range(seriesLength)], lengths+20+seriesLength),
                       ([(lengths+200)+x for x in range(seriesLength)], lengths+200+seriesLength)
        ]

        trainRegularLSTM(seriesLin, seriesLength, 1, 200, testSeries)
        trainStackedLSTM(seriesLin, seriesLength, 1, 200, 2, testSeries)
        #trainStackedLSTM(seriesLin, seriesLength, 1, 200, 3, testSeries)
        trainBidirectionalLSTM(seriesLin, seriesLength, 1, 200, testSeries)
    if False:
        logging.info("Predicting next squared value")
        seriesLength = 4    
        lengths = 200
        startVal = 2
        seriesSq = [x*x for x in range(1, lengths+1)]
        testSeries = [ ([x*x for x in range(lengths+3, lengths+3+seriesLength)], (lengths+3+seriesLength+1)*(lengths+3+seriesLength+1)),
                       ([x*x for x in range(lengths+30, lengths+30+seriesLength)], (lengths+30+seriesLength+1)*(lengths+30+seriesLength+1))]
        trainRegularLSTM(seriesSq, seriesLength, 1, 200, testSeries)
        trainStackedLSTM(seriesSq, seriesLength, 1, 200, 2, testSeries)
        #trainStackedLSTM(seriesSq, seriesLength, 1, 200, 3, testSeries)
        trainBidirectionalLSTM(seriesSq, seriesLength, 1, 200, testSeries)
