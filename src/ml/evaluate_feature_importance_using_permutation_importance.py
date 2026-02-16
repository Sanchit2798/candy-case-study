from sklearn.inspection import permutation_importance
from matplotlib import pyplot
import pandas as pd


class EvaluateFeatureImportanceUsingPermutationImportance:

    def __init__(self, random_state = 123) -> None:
        self.random_state = random_state

    def evaluate(self, model, data, features, target, scoring):
        results = permutation_importance(model, data[features], data[target], scoring= scoring, random_state = self.random_state)
        importance = results.importances_mean
        return pd.DataFrame({'Features': features, 'Importance score' : importance})

    def visualise_importance(self, importance_df, sort=False):
        if sort:
            importance_df = importance_df.sort_values(by="Importance score", ascending=False)
        importance_df.plot.bar(x="Features", y="Importance score")