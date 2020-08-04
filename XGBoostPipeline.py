import pandas as pd
from sklearn.pipeline import Pipeline

import numpy as np

#Custom pipeline imports
from featurepipeline import FeaturePruner
from featurepipeline import CategoricalTransformer
from featurepipeline import CustomImputer
from featurepipeline import OrdinalTransformer
from featurepipeline import FeatureCreator  
from featurepipeline import OutlierPruner
from featurepipeline import PriceSplitter 
from featurepipeline import FeatureSelector
from featurepipeline import FeatureTransformer

from sklearn.preprocessing import MinMaxScaler


#This is the pipeline we use for partially cleansing the data. The remaining
#pipeline continues below
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

X_train = numerical_pipeline.fit_transform(raw_train)

#Remaining feature pipeline
OP = OutlierPruner(train_data=True)
PS = PriceSplitter(train_data=True)
FS = FeatureSelector(train_data=True, corr_val = 0.05)
FT = FeatureTransformer(trans='box')
MM = MinMaxScaler()

#Final fitting of data
X_train = OP.fit_transform(X_train)
X_train,feature_select = FS.fit_transform(X_train)
X_train,y_train = PS.fit_transform(X_train)
X_train= FT.fit_transform(X_train)
X_train = MM.fit_transform(X_train)


print("We are using a total of",len(feature_select),"features.")


#Loading test data
file_path = 'test_fix.csv'
raw_test = pd.read_csv(file_path)
X_submission_data = numerical_pipeline.fit_transform(raw_test)


OP = OutlierPruner(train_data=False)
PS = PriceSplitter(train_data=False)
FS = FeatureSelector(train_data=False, corr_val = 0.0, features=feature_select)
FT = FeatureTransformer(trans='box')

X_submission_data = OP.fit_transform(X_submission_data)

X_submission_data = FS.fit_transform(X_submission_data)
X_submission_data,Id_df = PS.fit_transform(X_submission_data)

X_submission_data= FT.fit_transform(X_submission_data)
X_submission_data = MM.transform(X_submission_data)




#Helper Functions
def submission_creator(model,name):
    
    prediction_array = model.predict(X_submission_data)

    housing_prices =  {'Id': Id_df, 'SalePrice':np.exp(prediction_array) }
    df = pd.DataFrame(housing_prices, columns = ['Id', 'SalePrice'])
    
    submission_title = 'Submission' + name
    df.to_csv(submission_title+ '.csv',index = False)
    print('Submission Created!')


def display_scores(scores):
    # print(scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

from sklearn.model_selection import cross_val_score

def model_scorer(model):
    scores = cross_val_score(model, X_train, y_train,
    scoring="neg_mean_squared_error", cv=5)
    
    rmse_scores = np.sqrt(-scores)
    display_scores(rmse_scores)
    
    
    
import xgboost as xgb

# params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,
#                 'max_depth': 5, 'alpha': 10}
XGB_reg = xgb.XGBRegressor(eval_metric = 'rmse',silent = True)


params_XGB = {'eta': 0.03, 'max_depth': 3, 'n_estimators': 800,'alpha': 0.001, 'colsample_bytree': 0.6, 'lambda': 0.001, 'subsample': 0.6}
XGB_reg.set_params(**params_XGB)
model_scorer(XGB_reg)

XGB_reg.fit(X_train,y_train)

submission_creator(XGB_reg,'_xgb')


























