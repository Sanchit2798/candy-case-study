from sklearn.gaussian_process import GaussianProcessRegressor
import xgboost
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, DotProduct
from src.ml.ml_model_experiment import MlModelExperiment

def optimise_hyper_parameters_xgboost(ml_model_experiment:MlModelExperiment,
                                    xg_param_grid_in = None)->MlModelExperiment:
    if xg_param_grid_in is None:
        xg_param_grid = {
                            'max_depth': [3, 5, 8],
                            'n_estimators': [10, 100],
                            'colsample_bytree': [0.3, 0.5, 0.8],
                            'lambda' : [0,0.1, 0.5, 1, 2],
                            'subsample' : [0.3, 0.5]
                        }
    else:
        xg_param_grid = xg_param_grid_in
    ml_model_experiment._get_best_hyperparameters(xg_param_grid)
    ml_model_experiment.model = xgboost.XGBRegressor(**ml_model_experiment.model_best_hyper_parameters)
    return ml_model_experiment

def optimise_hyper_parameters_gpr(ml_model_experiment:MlModelExperiment,
                                gpr_param_grid_in = None)->MlModelExperiment:

    if gpr_param_grid_in is None:
        kernels = []
        for ck1 in range(1,2):   #tried [0.01, 0.1, 1,2]
            for rbf in range(5,6): # tried [0.01,0.1,1,2]
                for wk in range(1,2): # tried [0.01,0.1,1,2]
                    kernel_ = ConstantKernel(constant_value = ck1, constant_value_bounds =(1e-4,1e7)) + RBF(length_scale = rbf,  length_scale_bounds=(1e-2,1e7))  + WhiteKernel(noise_level = wk, noise_level_bounds=(1e-10,1e7))
                    kernels.append(kernel_)
                    
        gpr_param_grid = { 
                            "kernel": kernels,
                            'random_state':[123], 
                            'n_restarts_optimizer':[10],
                            'alpha':[0.01, 0.1], 
                            'normalize_y':[True]
                        } 
    else:
        gpr_param_grid = gpr_param_grid_in
    ml_model_experiment._get_best_hyperparameters(gpr_param_grid)
    ml_model_experiment.model = GaussianProcessRegressor(**ml_model_experiment.model_best_hyper_parameters)
    return ml_model_experiment