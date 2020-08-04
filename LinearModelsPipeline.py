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

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler


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
FT = FeatureTransformer()
MM = MinMaxScaler()
SS = StandardScaler()
RS = RobustScaler()
FS = FeatureSelector(train_data=True, corr_val = 0.05)
DHC = DropHighCorr(train_data=True, threshold = 0.80)


X_train = OP.fit_transform(X_train)
X_train,feature_select = FS.fit_transform(X_train)


X_train,y_train = PS.fit_transform(X_train)
X_train= FT.fit_transform(X_train)


# import matplotlib.pyplot as plt
# import seaborn as sns

# corrMatrix = X_train.corr()
# sns.heatmap(corrMatrix, annot=True)
# plt.show()


X_train,feature_highcorr = DHC.fit_transform(X_train)



column_vals = X_train.columns


X_train= SS.fit_transform(X_train)
X_train = MM.fit_transform(X_train)
# X_train = RS.fit_transform(X_train)



#Loading test data
file_path = 'test_fix.csv'
raw_test = pd.read_csv(file_path)

X_submission_data = numerical_pipeline.transform(raw_test)

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

X_submission_data= SS.transform(X_submission_data)
X_submission_data = MM.transform(X_submission_data)
# X_submission_data = RS.transform(X_submission_data)






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
    scoring="neg_mean_squared_error", cv=10)
    rmse_scores = np.sqrt(-scores)
    display_scores(rmse_scores)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet


from sklearn.model_selection import GridSearchCV


params= {"alpha": np.linspace(0.0002,50,200)}


lr_lasso = Lasso(max_iter=10000)
lasso_grid = GridSearchCV(lr_lasso, param_grid=params, scoring = 'neg_root_mean_squared_error', cv=5,n_jobs=-1)
# #Evaluating 
lasso_grid.fit(X_train, y_train)
print(lasso_grid.best_params_)
lr_lasso.set_params(**lasso_grid.best_params_)
model_scorer(lr_lasso)
lr_lasso.fit(X_train, y_train)
submission_creator(lr_lasso,'_lasso')



import matplotlib.pyplot as plt

# coefs = pd.Series(lr_lasso.coef_, index = column_vals)
# print("Lasso picked " + str(sum(coefs != 0)) + " features and eliminated the other " +  \
#       str(sum(coefs == 0)) + " features")
# imp_coefs = pd.concat([coefs.sort_values().head(10),
#                      coefs.sort_values().tail(10)])
# imp_coefs.plot(kind = "barh")
# plt.title("Coefficients in the Lasso Model")
# plt.show()







lr_ridge = Ridge(max_iter=10000)
ridge_grid = GridSearchCV(lr_ridge, param_grid=params, scoring = 'neg_root_mean_squared_error', cv=5,n_jobs=-1)
# #Evaluating 
ridge_grid.fit(X_train, y_train)
print(ridge_grid.best_params_)
lr_ridge.set_params(**ridge_grid.best_params_)
model_scorer(lr_ridge)
lr_ridge.fit(X_train, y_train)
submission_creator(lr_ridge,'_ridge')


from sklearn.ensemble import StackingRegressor



estimators = [
    ('lasso', Lasso(alpha = 0.0002,max_iter=10000)),
    ('ridge',Ridge(alpha = 1.2564763819095477,max_iter=10000))    
     ]
    
    
reg = StackingRegressor(estimators=estimators)
model_scorer(reg)
reg.fit(X_train, y_train)
submission_creator(reg,'_RidgeLassoStack')

from sklearn.ensemble import VotingRegressor

vot = VotingRegressor(estimators=estimators)
model_scorer(vot)
vot.fit(X_train, y_train)
submission_creator(vot,'_RidgeLassoAverage')










