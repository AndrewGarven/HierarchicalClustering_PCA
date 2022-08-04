import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def PCA_analysis(dataframe, feature_of_interest):
    # INPUT
    # dataframe = pandas dataframe of int or float values + has feature of interest (can be float, string, etc)
    # feature of interest = a string of the name of feature used for colour mapping (dataframe column heading)

    # OUTPUT
    # Matplotlib figure object containing a 2D PCA plot

    # first seperate the feature data from outcome data
    df = dataframe.drop(f'{feature_of_interest}', 1)
    outcome = dataframe.loc[:, f'{feature_of_interest}'].values.tolist()
    # scale the data so that each feature has equal weight in the PCA analysis
    x = StandardScaler().fit_transform(df)
    # PCA will reduce the entire dimensional space to a space of n_components (in most cases 2)
    pca = PCA(n_components=2)
    # fit the scaled data to 2 dimensions
    principalComponents = pca.fit_transform(x)
    # make a Pandas dataframe from the array output by PCA analysis
    principalDf = pd.DataFrame(data=principalComponents,
                               columns=['principal component 1', 'principal component 2'],
                               index=outcome)

    # create matplotlib plot features
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize =15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('', fontsize=20)

    # make data for scatter plot [x, y, colour]
    ax.scatter(principalDf.loc[:, 'principal component 1'],
               principalDf.loc[:, 'principal component 2'],
               c=principalDf.index.values.tolist())

    return fig.show()