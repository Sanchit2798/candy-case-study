from typing import List
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from typing import Tuple
from matplotlib import pyplot as plt


class PlotlyLib:
    
    @staticmethod
    def scatter_df(data : pd.DataFrame, x_col : str, y_col: str,
                error_x_col = None, 
                error_y_col = None,
                color_col = None, 
                size = None,
                hover_data_cols = None, 
                animation_frame = None, 
                animation_group=None,
                range_x=None, range_y=None,
                plot_title = None,
                show = True):
        if hover_data_cols is None:
            hover_data_cols = list(data.columns)
        if type(size) == int:
            size_col = None
        else:
            size_col = size 
        fig = px.scatter(data, x=x_col, y=y_col, color=color_col, hover_data=hover_data_cols, 
                        size= size_col, animation_frame=animation_frame, animation_group= animation_group, 
                        error_x= error_x_col, error_y= error_y_col, range_x=range_x, range_y=range_y)
        if type(size) == int:
            fig.update_traces(marker=dict(size=size),
                            selector=dict(mode='markers'))

        fig.update_layout(
                            title=plot_title
                        )
        if show:
            fig.show()
        return(fig)

    @staticmethod
    def add_series_arr(fig = None, x = None, y = None, error_x = None, error_y = None, mode = 'markers', name = None,
                        show = True, plot_title = None, xlabel = None, ylabel = None, width = 1000, height = 800):
        if fig is None:
            fig = go.Figure()
        fig.add_trace(go.Scatter(x = x, y = y, mode=mode, error_x =error_x, error_y = dict(type='data', array = error_y), name = name))
        fig.update_layout(
                            title=plot_title,
                            xaxis_title=xlabel,
                            yaxis_title=ylabel,
                            height = height,
                            width = width,
                            legend_title="Legend Title",
                            font=dict(
                                family="Courier New, monospace",
                                size=18,
                                color="RebeccaPurple",
                                
                            )
                            
                        )
        if show:
            fig.show()
        return(fig)

    @staticmethod
    def add_line(fig = None, x=None, y=None):
        if fig is None:
            fig = go.Figure()
        fig.add_trace(go.Scatter(x = x, y = y, mode='lines'))
        return(fig)

    @staticmethod
    def plot_error_two_dfs(data1:pd.DataFrame, data2:pd.DataFrame, 
                        x_axis:str, y_axis:str, x_err_= 'zero', y_err_ = 'zero',
                        x_lim = (0,100), y_lim = (0,100), data1_label = None,
                        data2_label = None, color1 = 'green', color2 = 'blue',
                        fig_size = (12, 10)):
        data1= data1.copy()
        data2 = data2.copy()
        data1['zero'] = 0
        data2['zero'] = 0
        plt.figure(figsize=fig_size)
        plt.errorbar(data1[x_axis], data1[y_axis], 
                    yerr=data1[y_err_], xerr=data1[x_err_], 
                    fmt='o', color = color1, label = data1_label, alpha = 0.8)
        plt.errorbar(data2[x_axis], data2[y_axis], 
                    yerr=data2[y_err_],  xerr=data2[x_err_],
                    fmt='o', color = color2, label = data2_label, alpha = 0.8)
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.xlim(x_lim[0],x_lim[1])
        plt.ylim(y_lim[0],y_lim[1])
        plt.legend()