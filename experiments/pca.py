from sklearn.decomposition import PCA
from data_utils import load_data
from sklearn.externals import joblib

# Builds a PCA model to reduce the dimensionality of
# the activations dataset. It saves the learnt model
# onto a file named "pca_model.pkl"


xs,ys = load_data("../activations.pkl")

data = [des for x in xs for des in x]

PCA_NUM_COMPONENTS = 50


pca = PCA(n_components=PCA_NUM_COMPONENTS)
pca.fit(data)


joblib.dump(pca, "pca_model.pkl")
