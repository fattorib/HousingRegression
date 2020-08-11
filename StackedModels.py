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

#Importing stacking regressor 
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import VotingRegressor

#Importing models to stack
import xgboost as xgb
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

#Setting parameters for all models (based on parameters found in other files)

XGB_reg = xgb.XGBRegressor(eval_metric = 'rmse',silent = True)
params_XGB = {'reg_alpha': 0.001, 'eta': 0.03, 'reg_lambda': 0.001, 'max_depth': 4, 'n_estimators': 1000, 'colsample_bytree': 0.6, 'subsample': 0.6}
XGB_reg.set_params(**params_XGB)


lr_lasso = Lasso(max_iter=10000,alpha = 0.0002)

lr_ridge = Ridge(max_iter=10000,alpha = 1.298710621242485)

#Creating stacked model 
estimators = [
    ('lasso', lr_lasso),
    ('xgb',XGB_reg),
    ('ridge',lr_ridge)
     ]
    
reg = StackingRegressor(estimators=estimators)
reg.fit(X_train, y_train)

#Creating submission
submission_creator(reg,'_RidgeXGBLassoStack')

#Creating averaged model
vot = VotingRegressor(estimators=estimators)

#Creating submission
vot.fit(X_train, y_train)
submission_creator(vot,'_RidgeXGBLassoAverage')
