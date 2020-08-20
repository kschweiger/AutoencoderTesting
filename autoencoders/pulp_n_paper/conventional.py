"""
Following: https://towardsdatascience.com/extreme-rare-event-classification-using-autoencoders-in-keras-a565b386f098
"""

import logging

import pandas as pd
import numpy as np

import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
#from tensorflow import set_random_seed

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score

import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff

from utils import initLogging


sign = lambda x : (1, -1)[x < 0]

def trainConventionalAutoencoder(data, seed=123, data_split=0.2, epochs=200, batchSize=128, encodingDim=32, hiddenDim=16, learningRate=1e-3 ):
    """
    Training of a conventional autoencoder on the pulp and paper dataset from ArXiv 1809.10717

    Args:
      
    """
    logging.warning("Start training")
    # Split passed data into training, testing and validation set
    data_train, data_test = train_test_split(data, test_size=data_split, random_state=seed)
    data_train, data_valid = train_test_split(data_train, test_size=data_split, random_state=seed)

    # Get Signal and background sets
    data_train_0 = data_train.loc[data_train['y'] == 0]
    data_train_1 = data_train.loc[data_train['y'] == 1]
    data_train_0_x = data_train_0.drop(['y'], axis=1)
    data_train_1_x = data_train_1.drop(['y'], axis=1)
    
    data_valid_0 = data_valid.loc[data_valid['y'] == 0]
    data_valid_1 = data_valid.loc[data_valid['y'] == 1]
    data_valid_0_x = data_valid_0.drop(['y'], axis=1)
    data_valid_1_x = data_valid_1.drop(['y'], axis=1)
    
    data_test_0 = data_test.loc[data_test['y'] == 0]
    data_test_1 = data_test.loc[data_test['y'] == 1]
    data_test_0_x = data_test_0.drop(['y'], axis=1)
    data_test_1_x = data_test_1.drop(['y'], axis=1)


    
    
    # Standardize to gauss w/ mu = 0 and sigma = 1
    # First get the transformation factors based on the trainig dataset
    scaler = StandardScaler().fit(data_train_0_x)

    # rescale training dataset, 
    data_train_0_x_rescaled = scaler.transform(data_train_0_x)
    # rescale background only validation dataset
    data_valid_0_x_rescaled = scaler.transform(data_valid_0_x)
    # rescale full validation dataset
    data_valid_x_rescaled = scaler.transform(data_valid.drop(['y'], axis = 1))

    logging.info("Number of training events (y == 0) : %s", len(data_train_0_x_rescaled))
    logging.info("Number of validation events (y == 0) : %s", len(data_valid_0_x_rescaled))
    logging.info("Number of validation events : %s", len(data_valid_x_rescaled))

    
    # same for testing
    data_test_0_x_rescaled = scaler.transform(data_test_0_x)
    data_test_x_rescaled = scaler.transform(data_test.drop(['y'], axis = 1))


    inputDim = data_train_0_x_rescaled.shape[1]
    # Now we get to the autoencoder
    input_layer = Input(shape=(inputDim, ))
    encoder = Dense(encodingDim, activation="relu", activity_regularizer=regularizers.l1(learningRate))(input_layer)
    encoder = Dense(hiddenDim, activation="relu")(encoder)
    decoder = Dense(hiddenDim, activation="relu")(encoder)
    decoder = Dense(encodingDim, activation="relu")(decoder)
    decoder = Dense(inputDim, activation="linear")(decoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.summary()

    input("Start training")

    autoencoder.compile(metrics=['accuracy'],
                        loss='mean_squared_error',
                        optimizer='adam')

    cp = ModelCheckpoint(filepath="autoencoder_classifier.h5",
                         save_best_only=True,
                         verbose=0)

    tb = TensorBoard(log_dir='./logs',
                     histogram_freq=0,
                     write_graph=True,
                     write_images=True)
    
    history = autoencoder.fit(data_train_0_x_rescaled, data_train_0_x_rescaled,
                              epochs=epochs,
                              batch_size=batchSize,
                              shuffle=True,
                              validation_data=(data_valid_0_x_rescaled, data_valid_0_x_rescaled),
                              verbose=1,
                              callbacks=[cp, tb]).history

    input("Find threshold")
    
    valid_x_predictions = autoencoder.predict(data_valid_x_rescaled)
    mse = np.mean(np.power(data_valid_x_rescaled - valid_x_predictions, 2), axis=1)
    error_df = pd.DataFrame({'Reconstruction_error': mse,
                            'True_class': data_valid['y']})
    # Get the precision and recall for the autoencoder.
    #   precision is tp / (tp + fp) --> tp : number of true positives / fp : number of false postives
    #     --> Ability of classifier not to label as positive a sample tat is negative
    #   recall is tp / (tp + fn) --> tp : number of true positives / fn : number of false negatives
    #     --> Ability of the classifier to find all the positive samples
    precision_rt, recall_rt, threshold_rt = precision_recall_curve(error_df.True_class, error_df.Reconstruction_error)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=precision_rt[1:], x=threshold_rt,
                             mode='lines',
                             name='precision'))
    fig.add_trace(go.Scatter(y=recall_rt[1:], x=threshold_rt,
                             mode='lines',
                             name='recall'))
    fig.update_layout(
        title="Precision and recall for different threshold values",
        xaxis_title="Threshold",
        yaxis_title="Precision/Recall"
    )
    fig.show()


    threshold_fixed = 0.4
    input("Evaluate testing sample with threshold %s"%threshold_fixed)

    test_x_predictions = autoencoder.predict(data_test_x_rescaled)
    mse = np.mean(np.power(data_test_x_rescaled - test_x_predictions, 2), axis=1)
    error_df_test = pd.DataFrame({'Reconstruction_error': mse,
                            'True_class': data_test['y']})
    error_df_test = error_df_test.reset_index()
    groups = error_df_test.groupby('True_class')
    fig = go.Figure()
    for name, group in groups:
        fig.add_trace(
            go.Scatter(
                x=group.index,
                y=group.Reconstruction_error,
                name="Break" if name == 1 else "Normal",
                mode='markers'
            )
        )
    fig.add_trace(
        go.Scatter(
            x=group.index,
            y=len(group.index)*[threshold_fixed],
            name="Threshold",
            mode='lines'
        )
    )
    
    fig.show()

    pred_y = [1 if e > threshold_fixed else 0 for e in error_df.Reconstruction_error.values]
    
    conf_matrix = confusion_matrix(error_df.True_class, pred_y)
    
    print(conf_matrix)
    logging.info("Predicted/Truth")
    logging.info("Normal/Normal : %s", conf_matrix[0][0])
    logging.info("Normal/Break : %s", conf_matrix[0][1])
    logging.info("Break/Normal : %s", conf_matrix[1][0])
    logging.info("Break/Break : %s", conf_matrix[1][1])
    logging.info("Brak predicted with %s", (conf_matrix[1][1]/(conf_matrix[1][0]+conf_matrix[1][1])))

    
    

    fig = ff.create_annotated_heatmap(z=conf_matrix,
                                      x =["Normal","Break"],
                                      y= ["Normal","Break"])
    
    fig['layout']['yaxis']['autorange'] = "reversed"
    fig.show()

    
    false_pos_rate, true_pos_rate, thresholds = roc_curve(error_df.True_class, error_df.Reconstruction_error)
    roc_auc = auc(false_pos_rate, true_pos_rate,)

    logging.info("ROC: %s", roc_auc)
    
    fig = go.Figure()
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    fig.add_trace(go.Scatter(x=false_pos_rate, y=true_pos_rate, name="AUC = {:.2f}".format(float(roc_auc)), mode='lines'))

    fig.update_layout(
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
    )
    fig.show()
    
def preProcessData(inputFile, shiftby = 2):
    """
    Read adn preprocess the dataset

    Args:
      inputFile (str) : Filename
    """
    logging.warning("Preprocessing data")
    logging.info("Got file: %s", inputFile)
    df = pd.read_csv(inputFile)
    #df = df[250:270] # FOR TESTING 
    # df is time series. Spacing is 2 minutes between rows
    
    # Autoencoder should be trained to predict the sheet break shiftby*2 minutes in advance
    # Therefore we want to the "positive" label up by shiftby and delete the actual "positive" one

    vector = df['y'].copy()
    initVector = vector.copy()
    for s in range(1,shiftby+1):
        tmp = initVector.shift(-1*s)
        tmp = tmp.fillna(0)
        vector += tmp
        
    df.insert(loc=0, column='y_tmp', value=vector)
    # Remove the rows with y == 1.
    df = df.drop(df[df["y"] == 1].index)
    df = df.drop("y", axis=1)
    df = df.rename(columns={'y_tmp': "y"})
    return df

if __name__ == "__main__":
    # from numpy.random import seed
    # seed(1)
    # from tensorflow import set_random_seed
    # set_random_seed(2)

    initLogging(20)
    
    traingData = preProcessData("dataset_1809.10717.csv")
    traingData = traingData.drop(['time', 'x28', 'x61'], axis=1)

    trainConventionalAutoencoder(traingData)
