import pandas as pd
from sklearn.pipeline import Pipeline

import numpy as np

#Custom pipeline imports
from featurepipeline import FeaturePruner
from featurepipeline import CategoricalTransformer
from featurepipeline import CustomImputer
from featurepipeline import OrdinalTransformer
from featurepipeline import FeatureCreator  
from featurepipeline import PolynomialFeatureCreator 
from featurepipeline import OutlierPruner
from featurepipeline import PriceSplitter 
from featurepipeline import FeatureSelector
from featurepipeline import FeatureTransformer
from featurepipeline import DropHighCorr

#Import feature scalers
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler



#Part of the pipeline is the same for both the train and test data, we initialize it here
numerical_pipeline = Pipeline([
('imputer', CustomImputer()),
('CategoricalEncoder', CategoricalTransformer()),
('OrdindalEncoder', OrdinalTransformer()),
('FeaturePruner', FeaturePruner()),
('FeatureCreation', FeatureCreator())
])

#Loading training data
file_path = 'train_fix.csv'
raw_train = pd.read_csv(file_path)

#Fitting and transforming data
X_train = numerical_pipeline.fit_transform(raw_train)


#Remaining feature pipeline, specific to training data
OP = OutlierPruner(train_data=True)
PS = PriceSplitter(train_data=True)
FT = FeatureTransformer()
FS = FeatureSelector(train_data=True, corr_val = 0.05)
DHC = DropHighCorr(train_data=True, threshold = 0.80)

#Initializing scalers
MM = MinMaxScaler()
SS = StandardScaler()

#Final fitting and transforming of training data
X_train = OP.fit_transform(X_train)
X_train,feature_select = FS.fit_transform(X_train)
X_train,y_train = PS.fit_transform(X_train)
X_train= FT.fit_transform(X_train)
X_train,feature_highcorr = DHC.fit_transform(X_train)

#Getting list of features
column_vals = X_train.columns

#Scaling data
X_train= SS.fit_transform(X_train)
X_train = MM.fit_transform(X_train)

#Loading test data
file_path = 'test_fix.csv'
raw_test = pd.read_csv(file_path)

#Fitting and transforming test data
X_submission_data = numerical_pipeline.transform(raw_test)

#Remaining feature pipeline, specific to test data
OP = OutlierPruner(train_data=False)
PS = PriceSplitter(train_data=False)
FT = FeatureTransformer()
FS = FeatureSelector(train_data=False, corr_val = 0.05, features=feature_select)
DHC = DropHighCorr(train_data=False,threshold = 0.80, features = feature_highcorr)

X_submission_data = OP.fit_transform(X_submission_data)
X_submission_data = FS.fit_transform(X_submission_data)
X_submission_data,Id_df = PS.fit_transform(X_submission_data)
X_submission_data= FT.fit_transform(X_submission_data)
X_submission_data = DHC.fit_transform(X_submission_data)

#Scaling data
X_submission_data= SS.transform(X_submission_data)
X_submission_data = MM.transform(X_submission_data)

#Importing helper functions for scoring and submission creation
from helper import submission_creator,display_scores,model_scorer


#Importing tree models
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
#We are using three different tree-based models: Random Forest, GradientBoostingRegressor, and XGBoost

#Using GridSearchCV to find optimal hyperparameters
from sklearn.model_selection import GridSearchCV

#1: Random Forest
rf = RandomForestRegressor(n_jobs = -1)

#Parameter space to check
params= {"criterion":['mae','mse'],"min_samples_split":[1,2,4], "max_features": ['auto', 'sqrt', 'log2'],
         "n_estimators": [100,200,400,800]}
rf_grid = GridSearchCV(rf, param_grid=params, scoring = 'neg_root_mean_squared_error', cv=5,n_jobs=-1)
rf_grid.fit(X_train, y_train)

#Output best parameters
print(rf_grid.best_params_)
rf.set_params(**rf_grid.best_params_)

#Scoring optimal parameters
model_scorer(rf)


#2: GradientBoostingRegressor
gb = GradientBoostingRegressor()

#Parameter space to check
params= {"loss":['ls', 'lad', 'huber'],  "criterion":['friedman_mse', 'mse'],"max_depth":[3,5] ,"min_samples_split":[1,4],"n_estimators": [200,400,800,1500], "learning_rate":[1,0.1,0.001]}
gb_grid = GridSearchCV(gb, param_grid=params, scoring = 'neg_root_mean_squared_error', cv=5,n_jobs=-1)
gb_grid.fit(X_train, y_train)

#Output best parameters
print(gb_grid.best_params_)
gb.set_params(**gb_grid.best_params_)

#Scoring optimal parameters
model_scorer(gb)



#3: XGBoost
XGB_reg = xgb.XGBRegressor(eval_metric = 'rmse')

#Pre-optimized parameters from previous GridSearchCV execution
params_opt = {'colsample_bytree': 0.6, 'subsample': 0.6}
XGB_reg.set_params(**params_opt)

#Parameter space to check
params= {'n_estimators':[400,800,1000,1500], 'max_depth':[3,4,5], 'eta':[0.30,0.03,0.003],'alpha':[0.001, 0.1, 0, 1],'lambda':[0.001, 0.1, 0, 1] }
XGB_grid = GridSearchCV(XGB_reg, param_grid=params, scoring = 'neg_root_mean_squared_error', cv=5,n_jobs=-1)
XGB_grid.fit(X_train, y_train)

#Output best parameters
print(XGB_grid.best_params_)
XGB_reg.set_params(**XGB_grid.best_params_)

#Scoring optimal parameters
model_scorer(XGB_reg)

