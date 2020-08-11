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

#Importing linear models
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge


#Finding optimal regularization parameter
from sklearn.model_selection import GridSearchCV
params= {"alpha": np.linspace(0.0002,50,200)}


lr_lasso = Lasso(max_iter=10000)
lasso_grid = GridSearchCV(lr_lasso, param_grid=params, scoring = 'neg_root_mean_squared_error', cv=10,n_jobs=-1)

#Model evaluation
lasso_grid.fit(X_train, y_train)
lr_lasso.set_params(**lasso_grid.best_params_)
model_scorer(lr_lasso)
lr_lasso.fit(X_train, y_train)

#Creating submission
submission_creator(lr_lasso,'_lasso')



lr_ridge = Ridge(max_iter=10000)
ridge_grid = GridSearchCV(lr_ridge, param_grid=params, scoring = 'neg_root_mean_squared_error', cv=10,n_jobs=-1)

#Model evaluation
ridge_grid.fit(X_train, y_train)
lr_ridge.set_params(**ridge_grid.best_params_)
model_scorer(lr_ridge)
lr_ridge.fit(X_train, y_train)

#Creating submission
submission_creator(lr_ridge,'_ridge')

