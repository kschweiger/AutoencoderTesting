import logging
import itertools

import pandas as pd
import numpy as np

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utils import initLogging

def plotDataset(datasetFile, batch=True):
    """
    Plot the dataset

    Args:
      datasetFile (str) : Path to dataset file
    """
    logging.info("Got file: %s", datasetFile)

    df = pd.read_csv(datasetFile)

    # Let's plot all features
    logging.info("Plotting features")
    figures = []
    for var in [x for x in df.columns if x not in ["time", "y", "x28", "x61"]]:
        figures.append(
            go.Scatter(x = df.time, y=df[var], name=var)
        )

    if not batch:
        fig = go.Figure(figures)
        fig.show()

    
    # PCA
    logging.info("Plotting Scree-plot")
    dfForPCA = df.copy()
    dfForPCA = dfForPCA.drop(["time", "y", "x28", "x61"], axis=1)
    logging.debug("N columns for PCA: %s", len(dfForPCA.columns))
    logging.debug("n Samples for PCA: %s", len(dfForPCA))
    dfForPCA = StandardScaler().fit_transform(dfForPCA)
    pca = PCA()
    pca.fit(dfForPCA)
    logging.debug("nComponents of PCA: %s", pca.n_components_)
    logging.debug("nFeatures of PCA: %s", pca.n_features_)
    logging.debug("nSamples of PCA: %s", pca.n_samples_)

    
    if not batch:
        fig = px.bar(pca.explained_variance_ratio_)
        fig.update_layout(
            title="Scree plot",
            xaxis_title="PCs",
            yaxis_title="Variance ratio"
        )
        fig.show()

    logging.info("PLotting cumulative variance")
    varSum = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3)*100)
    if not batch:
        fig = px.line(varSum)
        fig.update_layout(
            title="Cummulative variance",
            xaxis_title="PCs",
            yaxis_title="% Variance explained"
        )
        fig.show()

    # Correlation
    logging.info("Plotting correlation matrix")
    dfForCorr = df.copy()
    dfForCorr = dfForCorr.drop(["time", "y", "x28", "x61"], axis=1)
    corr = dfForCorr.corr()


    if not batch:
        fig = go.Figure(
            data=[
                go.Heatmap(z=corr,
                           x = corr.columns,
                           y= corr.columns,
                           colorscale=[[0, 'rgb(12,51,131)'],
                                       [0.25, 'rgb(10,136,186)'],
                                       [0.5, 'rgb(242,211,56)'],
                                       [0.75, 'rgb(242,143,56)'],
                                       [1, 'rgb(217,30,30)']]
                )
            ]
        )
        fig.update_layout(
            title="Feature Correlation"
        )
        fig.show()

    logging.info("Plotting 2D correlations")
    dfFor2DCorr = df.copy()
    dfFor2DCorr = dfFor2DCorr.drop(["time", "y", "x28", "x61"], axis=1)
    dfFor2DCorr = pd.DataFrame(StandardScaler().fit_transform(dfFor2DCorr),columns = dfFor2DCorr.columns)
    figures = []
    figures_titles = []
    for var1, var2 in itertools.combinations(list(dfFor2DCorr.columns), 2):
        logging.debug("Correlation for %s / %s", var1, var2)
        figures.append(
            go.Histogram2d(
                x = dfFor2DCorr[var1],
                y = dfFor2DCorr[var2]
            )
        )
        figures_titles.append((var1, var2))

    SubplotsPerPlot = (3,3)

    nSubplotsPerPlot = SubplotsPerPlot[0]*SubplotsPerPlot[1]
    nTotal = len(dfFor2DCorr.columns)

    
    if round(nTotal/nSubplotsPerPlot) < nTotal/nSubplotsPerPlot:
        nPlots = round(nTotal/nSubplotsPerPlot)+1
    else:
        nPlots = round(nTotal/nSubplotsPerPlot)
        
    logging.debug("Subplots : %s", SubplotsPerPlot)
    logging.debug("nTotal : %s", nTotal)
    logging.debug("nPlosts : %s", nPlots)

        
    iPLotted = 0
    for i in range(nPlots):
        fig = make_subplots(SubplotsPerPlot[0],SubplotsPerPlot[1])
        for ir in range(SubplotsPerPlot[0]):
            for ic in range(SubplotsPerPlot[1]):
                logging.debug("Plot %s -- iPLotted = %s / row = %s / column = %s", i, iPLotted, ir+1 ,ic+1)
                fig.add_trace(figures[iPLotted], row=ir+1 , col=ic+1)
                fig.update_xaxes(title_text=figures_titles[iPLotted][0], row=ir+1, col=ic+1)
                fig.update_yaxes(title_text=figures_titles[iPLotted][1], row=ir+1, col=ic+1)
                iPLotted += 1
        if not batch:
            fig.show()


if __name__ == "__main__":
    initLogging(20)
    
    dataset = "dataset_1809.10717.csv"
    
    plotDataset(dataset, batch=False)
