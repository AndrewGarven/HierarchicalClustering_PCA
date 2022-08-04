from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from PCA_plot import PCA_analysis
from matplotlib import pyplot as plt
import pandas as pd

path_to_file = 'dummyData.csv'

df = pd.read_csv(path_to_file)

def plotdendrogram(dataframe, linkage_type, feature_of_interest):
    # INPUT
    # dataframe = pandas dataframe of int or float values + has feature of interest (can be float, string, etc)
    # linkage_type = string containing dendogram linkage type (more info: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html)
    # feature of interest = a string of the name of feature used for colour mapping (dataframe column heading)

    # OUTPUT
    # Matplotlib figure object containing a 2D dendrogram object

    # remove feature of interest from feature dataframe
    df = dataframe.drop(f'{feature_of_interest}', 1)
    # isolate feature of interest and convert to numpy array
    outcome = dataframe.loc[:, f'{feature_of_interest}'].to_numpy()

    # calculate heirarchical clustering linkage (output: numpy array)
    linked = linkage(df, linkage_type)

    # set matplotlib figure size parameter
    plt.figure(figsize=(10,7))
    # create a top down dendrogram with feature of interest label
    dendrogram(linked,
               orientation='top',
               labels=outcome,
               distance_sort='descending',
               show_leaf_counts=True)
    return plt.show()

def plotClusteringPCA(dataframe, cluster_count, affinity, linkage_type, feature_of_interest):
    # INPUT
    # dataframe = (pandas dataframe) of int or float values + has feature of interest (can be float, string, etc)
    # cluster_count = (int) number of clusters outlined in dendrogram
    # affinity = (string) method of distance calculation often 'euclidean'
    # linkage_type = string containing dendogram linkage type (more info: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html)
    # feature of interest = a string of the name of feature used for colour mapping (dataframe column heading)

    # OUTPUT
    # Matplotlib figure object containing a 2D PCA plot with heirarchical clustering labels

    # remove feature of interest from feature dataframe
    df = dataframe.drop(f'{feature_of_interest}', 1)

    # Recursively merges pair of clusters of sample data; uses linkage distance
    cluster = AgglomerativeClustering(n_clusters=cluster_count, affinity=affinity, linkage=linkage_type)
    # Fit and return the result of each sample’s clustering assignment
    cluster.fit_predict(df)
    # add column to feature dataframe with clustering labels
    df['heirarchical_cluster'] = cluster.labels_

    return(PCA_analysis(df, 'heirarchical_cluster'))