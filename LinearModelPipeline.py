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
FS = FeatureSelector(train_data=True, corr_val = 0.3)
FT = FeatureTransformer()
MM = MinMaxScaler()

#Final fitting of data
X_train = OP.fit_transform(X_train)
X_train,feature_select = FS.fit_transform(X_train)
X_train,y_train = PS.fit_transform(X_train)
X_train= FT.fit_transform(X_train)
X_train = MM.fit_transform(X_train)


#Loading test data
file_path = 'test_fix.csv'
raw_test = pd.read_csv(file_path)
X_submission_data = numerical_pipeline.fit_transform(raw_test)

OP = OutlierPruner(train_data=False)
PS = PriceSplitter(train_data=False)
FS = FeatureSelector(train_data=False, corr_val = 0.3, features=feature_select)
FT = FeatureTransformer()


X_submission_data = OP.fit_transform(X_submission_data)
X_submission_data = FS.fit_transform(X_submission_data)
X_submission_data,Id_df = PS.fit_transform(X_submission_data)

X_submission_data= FT.fit_transform(X_submission_data)
X_submission_data = MM.transform(X_submission_data)



