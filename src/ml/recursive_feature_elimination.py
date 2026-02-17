
from typing import List
import pandas as pd
from src.ml.train_evaluate_save_ml_model import TrainEvaluateSaveMlModel
import copy
import numpy as np
import logging

# Configure the logging module
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class RecursiveFeatureElimination:

    def __init__(self, ml_model, data:pd.DataFrame, feature_list:List, fixed_feature_list : List, 
                target, min_features_num = 1,
                standardisation = True,
                test_size = 0.2,
                optmise_hyperparameters_for_models = True,
                score = 'R2 score',
                thresh = None,
                verbose=False):

        self.ml_model = ml_model
        self.data = data
        self.feature_list = feature_list
        self.fixed_feature_list = copy.deepcopy(fixed_feature_list)
        self.target = target
        self.standardisation = standardisation
        self.min_features_num = min_features_num
        self.test_size = test_size
        self.optmise_hyperparameters_for_models = optmise_hyperparameters_for_models
        self.score = score
        self.features_for_each_iteration = []
        self.scores_for_each_iteration = []
        self.feature_removed_for_each_iteration = []
        self.score_list_with_each_feature_removed_for_each_iteration = []
        self.thresh = thresh
        self.verbose = verbose


    def run(self, k_fold = 5):
        self.current_feature_list = copy.deepcopy(self.feature_list)
        self.run_model_experiment(k_fold, self.current_feature_list)

        while len(self.current_feature_list) > self.min_features_num:
            if self.verbose:
                logging.info("\n number of features left: " + str(len(self.current_feature_list)) + "\n")

            score_list_with_each_feature_removed = []
            for feature in self.current_feature_list:
                
                features = self.current_feature_list[:]
                features.remove(feature)

                modelResults = self.run_model_experiment(k_fold, features + self.fixed_feature_list)
                score_list_with_each_feature_removed.append(modelResults[modelResults['metric'] == self.score]['average'].values[0])
                self.score_list_with_each_feature_removed = score_list_with_each_feature_removed
            
            removed_feature = self.get_removed_feature(score_list_with_each_feature_removed)

            self.features_for_each_iteration.append(list(self.current_feature_list + self.fixed_feature_list))
            self.scores_for_each_iteration.append(max(score_list_with_each_feature_removed))
            self.feature_removed_for_each_iteration.append(removed_feature)
            self.score_list_with_each_feature_removed_for_each_iteration.append(score_list_with_each_feature_removed)
        return self.features_for_each_iteration, self.scores_for_each_iteration, self.feature_removed_for_each_iteration, self.score_list_with_each_feature_removed_for_each_iteration


    def get_removed_feature(self, score_list_with_each_feature_removed):
        if self.thresh is None:
            removed_feature_index = score_list_with_each_feature_removed.index(max(score_list_with_each_feature_removed))
            removed_feature = [self.current_feature_list.pop(removed_feature_index)]
        else:
            removed_feature = []
            removed_feature_indices = []
            for feat_score in score_list_with_each_feature_removed:
                max_score = max(score_list_with_each_feature_removed)
                if (abs((feat_score) - max_score)< abs(self.thresh*max_score)) & (len(self.current_feature_list) > self.min_features_num):
                    removed_feature_index = score_list_with_each_feature_removed.index(feat_score)
                    removed_feature_indices.append(removed_feature_index)
                    removed_feature.append(self.current_feature_list[removed_feature_index])
            
            self.current_feature_list = list(np.delete(np.array(self.current_feature_list), tuple(removed_feature_indices)))
        return removed_feature


    def run_model_experiment(self, k_fold, features):
        tesm = TrainEvaluateSaveMlModel(features=features,
                                 target=self.target, 
                                 data=self.data, 
                                 test_size=self.test_size,
                                 standardisation=self.standardisation,
                                 k_fold=k_fold,
                                 plotting=False,
                                 optimise_hyper_parameters=self.optmise_hyperparameters_for_models,
                                 verbose=self.verbose)
        
        if type(self.ml_model) == str:
           ml_expriment = tesm.run(self.ml_model)
        else:
            ml_expriment = tesm.run_with_my_model(self.ml_model)
        
        return ml_expriment.cv_results