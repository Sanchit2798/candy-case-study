import itertools
import numpy as np
import pandas as pd
from typing import List, Dict
from itertools import islice
import math

class FullFactorialDesignSpaceGenerator:

    def __init__(self, features : List[str], feature_max_dict: Dict[str, float], feature_min_dict: Dict[str,float], 
                 feature_step_dict: Dict[str,float]) -> None:
        self.features = features
        self.feature_max_dict = feature_max_dict
        self.feature_min_dict = feature_min_dict
        self.design_space_feature_lists = [list(np.arange(feature_min_dict[feature], 
                                                          feature_max_dict[feature]+feature_step_dict[feature], 
                                                          feature_step_dict[feature])) for feature in features]

    def compute_design_space_size(self):
        self.feat_lens = [len(x) for x in self.design_space_feature_lists]
        return math.prod(self.feat_lens)

    def create_design_space_iterator(self):
        return itertools.product(*self.design_space_feature_lists)

    def generate_design_space_df(self):
        design_space = list(self.create_design_space_iterator())
        design_space = pd.DataFrame(design_space, columns=self.features)
        return design_space