import os
import sys
from dataclasses import dataclass
from src.utils import evaluate_models

from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
        
    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("splitting training and test input data")
            X_train=train_arr[:,:-1]
            Y_train=train_arr[:,-1]
            X_test=test_arr[:,:-1]
            Y_test=test_arr[:,-1]
            
            
            models={
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "linear Regressor":LinearRegression(),
                "K-Neighbors":KNeighborsRegressor(),
                "XGBoost":XGBRegressor(),
                "CatBoosting":CatBoostRegressor(verbose=False),
                "AdaBoost":AdaBoostRegressor()
            
                
            }
            model_report:dict=evaluate_models(X_train=X_train,Y_train=Y_train,X_test=X_test,Y_test=Y_test,models=models )
            
            # best score
            best_model_score=max(sorted(model_report.values()))
            
            # best model
            best_model_score = max(model_report.values())

            best_model_name = max(model_report, key=model_report.get)

            best_model = models[best_model_name]
            
            if best_model_score <0.6:
                raise CustomException("no best model found")
            logging.info(f" best found model on both training dataset")
            
            save_object( 
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            best_model.fit(X_train, Y_train)
            
            predicted=best_model.predict(X_test)
            
            r2=r2_score(Y_test,predicted)
            
            return r2
        except Exception as e: 
            raise CustomException(e,sys)
                