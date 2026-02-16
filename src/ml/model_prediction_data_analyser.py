from typing import Dict
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from src.ml.plotting.plotly_lib import PlotlyLib

class ModelPredictionDataAnalyser:
    def __init__(self, actual, predicted, predicted_std):
        self.actual = actual
        self.predicted = predicted
        self.predicted_std = predicted_std
    
    def plot_prediction_data(self, fig, max_y, min_y, is_regression=True):
        fig = PlotlyLib.add_line(fig, x = [min_y, max_y], y = [min_y, max_y])
        fig.show()

    def get_metrics(self, y_actual_, y_pred, is_regression=True):
        metrics_result_dict = {}
        if is_regression:
            metrics_result_dict['r2 score'] = r2_score(y_actual_, y_pred)
            metrics_result_dict['mae - '] = mean_absolute_error(y_actual_, y_pred)
        else:
            metrics_result_dict['accuracy'] = accuracy_score(y_actual_, y_pred) 
        return metrics_result_dict


class ModelPredictionAnalyserArrays(ModelPredictionDataAnalyser):
    def __init__(self, actual:np.array, predicted:np.array, predicted_std=None) -> None:
        super().__init__(actual, predicted, predicted_std)

    def plot_prediction_data(self,
                    x_label = 'actual', y_label = 'predicted',
                    plot_title = None,
                    color = None, is_regression=True):
        
        fig = PlotlyLib.add_series_arr(x=self.actual, y=self.predicted, error_y=self.predicted_std,
                                    xlabel=x_label, ylabel=y_label, plot_title= plot_title, show=False)
        max_y = max(np.append(self.actual, self.predicted))
        min_y = min(np.append(self.actual, self.predicted))
        super().plot_prediction_data(fig, max_y, min_y, is_regression=is_regression)
    
    def get_metrics(self, is_regression=True):
        return super().get_metrics(self.actual, self.predicted, is_regression=is_regression)


class ModelPredictionAnalyserDfs(ModelPredictionDataAnalyser):
    def __init__(self, df:pd.DataFrame, actual:str, predicted:str, predicted_std= None) -> None:
        super().__init__(actual, predicted, predicted_std)
        self.df = df

    def plot_prediction_data(self,
                    plot_title = None,
                    color_col = None,
                    width_fig = 1000,
                    height_fig = 800,
                    is_regression=True):
        data = self.df

        if color_col is not None:
            data = data.sort_values(by=color_col, ascending=False)

        fig = PlotlyLib.scatter_df(data=data, x_col=self.actual,
                                y_col=self.predicted, error_y_col=self.predicted_std, 
                                color_col=color_col, show=False,
                                plot_title = plot_title)
        fig.update_layout(
                        autosize=False,
                        width=width_fig,
                        height=height_fig)

        max_y = max(self.df[[self.actual, self.predicted]].max(axis=1))
        min_y = min(self.df[[self.actual, self.predicted]].min(axis=1))
        super().plot_prediction_data(fig, max_y, min_y, is_regression=is_regression)

    def to_excel(self, name:str):
        self.df.to_excel(name, engine='openpyxl')

    def get_metrics(self, is_regression=True):
        return super().get_metrics(self.df[self.actual], self.df[self.predicted], is_regression=is_regression)


class TrainTestModelPredictionDataMerger():

    def __init__(self, train_prediction_data: ModelPredictionAnalyserDfs, 
                test_prediction_data: ModelPredictionAnalyserDfs):
            self.train_data = train_prediction_data
            self.test_data = test_prediction_data
            self.train_data.df.loc[:, 'Train set'] = '1'
            self.test_data.df.loc[:, 'Train set'] = '0'
            full_data = pd.concat([self.train_data.df, self.test_data.df])
            self.full_data = ModelPredictionAnalyserDfs(full_data, self.train_data.actual,
                                                        self.train_data.predicted,
                                                        self.train_data.predicted_std)
    def full_to_excel(self, name:str):
        self.full_data.df.to_excel(name, engine='openpyxl')

    def full_to_csv(self, name:str):
        self.full_data.df.to_csv(name)

    def plot_prediction_data(self,
                    plot_title = None,
                    color_col = 'Train set',
                    width_fig = 1000,
                    height_fig = 800):
        data = self.full_data.df
        if color_col is not None:
            data = data.sort_values(by=color_col, ascending=False)

        fig = PlotlyLib.scatter_df(data=data, x_col=self.full_data.actual,
                                y_col=self.full_data.predicted, error_y_col=self.full_data.predicted_std, 
                                color_col=color_col, show=False,
                                plot_title = plot_title)
        fig.update_layout(
                        autosize=False,
                        width=width_fig,
                        height=height_fig)

        max_y = max(self.full_data.df[[self.full_data.actual, self.full_data.predicted]].max(axis=1))
        min_y = min(self.full_data.df[[self.full_data.actual, self.full_data.predicted]].min(axis=1))
        fig = PlotlyLib.add_line(fig, x = [min_y, max_y], y = [min_y, max_y])
        fig.show()