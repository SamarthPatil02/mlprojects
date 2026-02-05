import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
         self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self,train_arr,test_arr):   ##we are passing parameters beacuse this functions belongs to output of model
        try:
            logging.info("Spliting training and testing input data")

            X_train = train_arr[:,:-1]
            y_train = train_arr[:,-1]
            X_test =   test_arr[:,:-1]
            y_test =   test_arr[:,-1]
             

            models = {
                "Linear regression" : LinearRegression(),
                "Adaboost":AdaBoostRegressor(),
                "Gradient" : GradientBoostingRegressor(),
                "Random Forest":RandomForestRegressor(),
                "decision tree" : DecisionTreeRegressor(),
                "KNN": KNeighborsRegressor(),
                "XGBoost":XGBRegressor()           
            }

            model_report = evaluate_models(X_train,X_test,y_train,y_test,models)
        
        
            #to get the best score from dict
            best_model_score = max(sorted(model_report.values()))

            #To get the best name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            
            logging.info("Best found model on training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model 
            )

            predicted = best_model.predict(X_test)

            score = r2_score(y_test, predicted)
            return score

        except Exception as e:
            raise CustomException(e,sys)