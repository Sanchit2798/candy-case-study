import pathlib
from src.ml.ml_model_experiment import MlModelExperiment
from src.ml.model_evaluator_on_split_data import ModelEvaluatorOnSplitData
from src.ml.model_prediction_data_analyser import ModelPredictionAnalyserDfs, TrainTestModelPredictionDataMerger

def analyse_ml_experiment_train_test_results_dfs(ml_model_experiment:MlModelExperiment):

    train_set = ml_model_experiment.training_data_set
    test_set = ml_model_experiment.testing_data_set
    train_set['Prediction'] = ml_model_experiment.model_evaluator.y_train_pred
    train_set['Prediction Std'] = ml_model_experiment.model_evaluator.y_train_pred_std

    test_set['Prediction'] = ml_model_experiment.model_evaluator.y_test_pred
    test_set['Prediction Std'] = ml_model_experiment.model_evaluator.y_test_pred_std

    predicted_train_set = ModelPredictionAnalyserDfs(train_set, ml_model_experiment.target,
                                                    'Prediction', 'Prediction Std')
    predicted_test_set = ModelPredictionAnalyserDfs(test_set, ml_model_experiment.target,
                                                    'Prediction', 'Prediction Std')
    return predicted_train_set, predicted_test_set


def plot_save_ml_experiment_train_test_results_dfs(ml_experiment:MlModelExperiment, prediction_data_file_name = None, plotting = True):
    predicted_train_data, predicted_test_data = analyse_ml_experiment_train_test_results_dfs(ml_experiment)
    predicted_data = TrainTestModelPredictionDataMerger(predicted_train_data, 
                                        predicted_test_data)
    if plotting:
        predicted_data.plot_prediction_data()
    if prediction_data_file_name is not None:
        file_extension = pathlib.Path(prediction_data_file_name).suffix
        if file_extension == '.xlsx':
            predicted_data.full_to_excel(prediction_data_file_name)
        else:
            predicted_data.full_to_csv(prediction_data_file_name)

