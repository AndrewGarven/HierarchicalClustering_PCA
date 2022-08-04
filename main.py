import pandas as pd
from HeirarchicalClustering import plotdendrogram, plotClusteringPCA
from PCA_plot import PCA_analysis

df = pd.read_csv('dummyData.csv')

PCA_analysis(df, 'output')
plotdendrogram(df, 'average', 'output')
plotClusteringPCA(df, 2, 'euclidean', 'average', 'output')

