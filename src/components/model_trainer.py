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
            params={
                "Random Forest":{
                    "n_estimators":[8,16,32,54,64,128,256]
                    },
                "Decision Tree":{
                    "criterion":["squared_error","friedman_mse","absolute_error","poisson"]
                     },
                
                "Gradient Boosting":{
                    "learning_rate":[.1,0.01,.001,.005,.5],
                    "subsample":[0.6,0.7,.75,.85,.8],
                    'n_estimators':[8,16,32,54,128,256]
                        },
                
                "XGBoost":{
                    "learning_rate":[.1,.001,.5,.005],
                    "n_estimators":[8,16,32,56,128,256]
                },
                
                'linear Regressor':{},
                "K-Neighbors":{
                    "n_neighbors":[5,7,9,11]
                },
                
                "CatBoosting":{
                    
                    "learning_rate":[.1,.001,.5,.005],
                    "depth":[6,8,9],
                    "iterations":[30,50,100]
                },
                'AdaBoost':{
                    "learning_rate":[.1,.001,.05,.5],
                    "n_estimators":[8,16,32,54,128,256]
                }
                

                
                
                
            }
                        
            model_report:dict=evaluate_models(X_train=X_train,Y_train=Y_train,X_test=X_test,Y_test=Y_test,models=models,param=params )
            
            # # best score
            # best_model_score=max(sorted(model_report.values()))
            
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
                