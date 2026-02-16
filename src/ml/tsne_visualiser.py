from sklearn.manifold import TSNE
from src.ml.standardise_train_test_data import standardise_data
import copy
import plotly.express as px

class TSNEVisualiser:

    def __init__(self) -> None:
        self.projection_x_name = 'projection_x'
        self.projection_y_name = 'projection_y'
        self.data = None

    def fit_data(self, data, possible_features,
                 n_components = 2, random_state = 0, n_iter_without_progress =300, max_iter = 500, 
                 n_job = -1, verbose = 2):
        standardised_data, standard_scaler = standardise_data(copy.deepcopy(data[possible_features]))
        tsne = TSNE(n_components=n_components, random_state=random_state, n_iter_without_progress=n_iter_without_progress, max_iter=max_iter, n_jobs=n_job, verbose=verbose)
        projections = tsne.fit_transform(standardised_data)

        print(projections.shape)
        data[self.projection_x_name] = projections[:, 0]
        data[self.projection_y_name] = projections[:, 1]
        self.data = data
        return data