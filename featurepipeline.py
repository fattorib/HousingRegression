
from sklearn.base import BaseEstimator, TransformerMixin

class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        print('Feature Pipeline!')
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X, y=None):

        neighbourhood_grouped_feat = ['Electrical', 'KitchenQual', 'Exterior1st', 
                                'Exterior2nd', 'Functional', 'MSZoning', 
                                'SaleType', 'Utilities']
        
        # for feature in neighbourhood_grouped_feat:
            # X[feature] = X.groupby(['Neighborhood', 'MSSubClass'])[feature].apply(lambda c: c.fillna(c.value_counts().index[0]))
           
        X['Electrical'] = X.groupby('Neighborhood')['Electrical'].transform(lambda x: x.fillna(x.mode()[0]))
        X['KitchenQual'] = X.groupby('Neighborhood')['KitchenQual'].transform(lambda x: x.fillna(x.mode()[0]))
        X['Exterior1st'] = X.groupby('Neighborhood')['Exterior1st'].transform(lambda x: x.fillna(x.mode()[0]))
        X['Exterior2nd'] = X.groupby('Neighborhood')['Exterior2nd'].transform(lambda x: x.fillna(x.mode()[0]))
        X['Functional'] = X.groupby('Neighborhood')['Functional'].transform(lambda x: x.fillna(x.mode()[0]))
        X['MSZoning'] = X.groupby('Neighborhood')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
        X['SaleType'] = X.groupby('Neighborhood')['SaleType'].transform(lambda x: x.fillna(x.mode()[0]))
        X['Utilities'] = X.groupby('Neighborhood')['Utilities'].transform(lambda x: x.fillna(x.mode()[0]))
        
        #Fill thse columns with 'None'

        general_missing_NA= ['FireplaceQu','PoolQC','Fence','MiscFeature',
                             'Alley']
        
        for feature in general_missing_NA:
            X[feature] = X[feature].fillna('NA')
        
        basement_missing_zero = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                                 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']
        
        for feature in basement_missing_zero:
            X[feature] = X[feature].fillna(0)
        
        
        basement_missing_none = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 
                                 'BsmtFinType1', 'BsmtFinType2']
        
        for feature in basement_missing_none:
            X[feature] = X[feature].fillna('NA')
        
        
        garage_missing_zero = ['GarageYrBlt', 'GarageArea', 'GarageCars']
        
        for feature in garage_missing_zero:
            X[feature] = X[feature].fillna(0)
        
        garage_missing_none = ['GarageType', 'GarageFinish', 'GarageQual', 
                               'GarageCond']
        
        for feature in garage_missing_none:
            X[feature] = X[feature].fillna('NA')
        
        
        
        
        X['MasVnrType'] = X['MasVnrType'].fillna('None')
        X['MasVnrArea'] = X['MasVnrArea'].fillna(0)
        
        X["LotFrontage"] = X.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
        
        return X   
        
        

class CategoricalTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self): # no *args or **kargs
        self.cat_list = ['GarageType','Alley','MSZoning','Street','LotShape','LandContour','Utilities',
                        'LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType',
                        'HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType',
                        'Foundation','Heating','Electrical','CentralAir','PavedDrive','SaleType',
                        'SaleCondition']
        
        self.Garage_dict = ['2Types','Attchd','Basment','BuiltIn','CarPort','Detchd','NA']
        
        self.alley_dict = ['Grvl','Pave','NA']
        
        self.MSZoning_dict=['A','C','FV','I','RH','RL','RP','RM','C (all)']
        
        self.Street_dict=['Grvl','Pave']
        
        self.LotShape_dict=['Reg','IR1','IR2','IR3']
        
        self.LandContour_dict=['Lvl','Bnk','HLS','Low']
        
        self.Utilities_dict=['AllPub','NoSewr','NoSeWa','ELO']
        
        self.LotConfig_dict=['Inside','Corner','CulDSac','FR2','FR3']
        
        self.LandSlope_dict=['Gtl','Mod','Sev']
        
        self.Neighborhood_dict=['Blmngtn','Blueste','BrDale','BrkSide','ClearCr',
                           'CollgCr','Crawfor','Edwards','Gilbert','IDOTRR',
                           'MeadowV','Mitchel','NAmes','NoRidge','NPkVill',
                           'NridgHt','NWAmes','OldTown','SWISU','Sawyer',
                           'SawyerW','Somerst','StoneBr','Timber','Veenker']
        
        self.Condition1_dict=['Artery','Feedr','Norm','RRNn','RRAn','PosN','PosA','RRNe','RRAe']
        
        self.Condition2_dict=['Artery','Feedr','Norm','RRNn','RRAn','PosN','PosA','RRNe','RRAe']
        
        self.BldgType_dict=['1Fam','2fmCon','Duplex','Twnhs','TwnhsE','TwnhsI']
        
        self.HouseStyle_dict=['1Story','1.5Fin','1.5Unf','2Story','2.5Fin','2.5Unf','SFoyer','SLvl']
        
        self.RoofStyle_dict=['Flat','Gable','Gambrel','Hip','Mansard','Shed']
        
        self.RoofMatl_dict=['ClyTile','CompShg','Membran','Metal','Roll','Tar&Grv','WdShake','WdShngl']
        
        self.Exterior1st_dict=['AsbShng','AsphShn','BrkComm','BrkFace','CBlock','CemntBd','HdBoard','ImStucc',
                          'MetalSd','Other','Plywood','PreCast','Stone','Stucco','VinylSd','Wd Sdng','WdShing']
        
        self.Exterior2nd_dict=['AsbShng','AsphShn','BrkComm','BrkFace','CBlock','CemntBd','HdBoard','ImStucc',
                          'MetalSd','Other','Plywood','PreCast','Stone','Stucco','VinylSd','Wd Sdng','WdShing']
        
        self.MasVnrType_dict=['BrkCmn','BrkFace','CBlock','None','Stone']
        
        self.Foundation_dict=['BrkTil','CBlock','PConc','Slab','Stone','Wood']
        
        self.Heating_dict=['Floor','GasA','GasW','Grav','OthW','Wall']
                         
        self.Electrical_dict=['SBrkr','FuseA','FuseF','FuseP','Mix']
                         
        self.CentralAir_dict = ['Y','N']
                         
        self.PavedDrive_dict = ['Y','P','N']
                         
        self.SaleType_dict=['WD','CWD','VWD','New','COD','Con','ConLw','ConLI','ConLD','Oth']
                         
        self.SaleCondition_dict=['Normal','Abnorml','AdjLand','Alloca','Family','Partial']
        
        self.all_dict = [self.Garage_dict,self.alley_dict,self.MSZoning_dict,self.Street_dict,self.LotShape_dict,self.LandContour_dict,self.Utilities_dict,self.LotConfig_dict,self.LandSlope_dict,
                   self.Neighborhood_dict,self.Condition1_dict,self.Condition2_dict,self.BldgType_dict,self.HouseStyle_dict,self.RoofStyle_dict,self.RoofMatl_dict,
                   self.Exterior1st_dict,self.Exterior2nd_dict,self.MasVnrType_dict,self.Foundation_dict,self.Heating_dict,self.Electrical_dict,self.CentralAir_dict,
                  self.PavedDrive_dict,self.SaleType_dict,self.SaleCondition_dict]
        
        self.list_keys = list(zip(self.cat_list,self.all_dict))
        # print('Using custom categorical transformer method')
        
    def fit(self, X, y=None):
        return self # nothing else to do
        
    def transform(self, X, y=None):
        from sklearn.preprocessing import OneHotEncoder
        import pandas as pd
        import numpy as np
        
        one_hot_df = pd.DataFrame(np.ones(1413),columns = ['tmp'])


        for feature,dictionary in self.list_keys:
            oh = OneHotEncoder()
            oh.fit(np.array(dictionary).reshape(-1, 1))
            OH_features = oh.transform(X[feature].values.reshape(-1, 1))
            OH_df = pd.DataFrame.sparse.from_spmatrix(OH_features,columns = dictionary)
            
            OH_df['tmp']=1
            one_hot_df['tmp']=1
            
            X = X.drop(feature,axis =1)
            one_hot_df = one_hot_df.join(OH_df,lsuffix="_left", rsuffix="_right")
            
            one_hot_df = one_hot_df.drop('tmp_right', axis=1)
            one_hot_df = one_hot_df.drop('tmp_left', axis=1)
        
        
        one_hot_df['tmp']=1
        
        X['tmp']=1
        
        
        X = X.join(one_hot_df,how = 'left', lsuffix="_left", rsuffix="_right")
        X = X.drop('tmp_right', axis=1)
        X = X.drop('tmp_left', axis=1)
        X = X.fillna(0)
        return X

    

class OrdinalTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self): # no *args or **kargs
        self.ordinal_features = ['YearBuilt','YearRemodAdd','OverallQual',
                                 'OverallCond','ExterQual','ExterCond',
                                 'BsmtQual','BsmtCond','BsmtExposure',
                    'HeatingQC','KitchenQual','Functional','FireplaceQu',
                    'GarageQual','GarageCond','PoolQC','BsmtFinType1',
                    'BsmtFinType2','Fence','GarageFinish']
        
    def fit(self, X, y=None):
        return self # nothing else to do
        
    def transform(self, X, y=None):        
        from sklearn.preprocessing import LabelEncoder

        for feature in self.ordinal_features:
            #initialize new instance of labelencoder
            le = LabelEncoder()
            le.fit(X[feature])
            X[feature] = le.transform(X[feature])
        return X
        

class FeaturePruner(BaseEstimator, TransformerMixin):
    
    def __init__(self): # no *args or **kargs
        self.features_to_drop = ['MSSubClass',
                                 'YrSold','MoSold','MiscFeature','PoolArea'
                                 ]

        # print('Dropping features')

        
    def fit(self, X, y=None):
        return self # nothing else to do
        
    def transform(self, X, y=None):        
        X = X.drop(self.features_to_drop,axis=1)
        return X


class FeatureCreator(BaseEstimator, TransformerMixin):
    
    def __init__(self): # no *args or **kargs
        print('Feature Pipeline!')

    def fit(self, X, y=None):
        return self # nothing else to do
        
    def transform(self, X, y=None):    
        import pandas as pd
        X['TotalSF'] = X['TotalBsmtSF'] + X['1stFlrSF'] + X['2ndFlrSF']
        X['PorchSF'] = X['OpenPorchSF'] + X['3SsnPorch'] + X['EnclosedPorch'] + X['ScreenPorch'] + X['WoodDeckSF']
        X['TotalBath'] = X['BsmtFullBath'] + X['FullBath'] + 0.5*X['BsmtHalfBath'] + 0.5*X['HalfBath']
        X['RemodSum']= X['YearRemodAdd'] + X['YearBuilt']
        # X['Bedrooms/RM']= X['BedroomAbvGr']/X['TotRmsAbvGrd']
        
        X['OverallCombined'] = X['OverallQual']+X['OverallCond']

        X['ExterCombined'] = X['ExterQual']+X['ExterCond']
        
        X['BasementCombined'] = X['BsmtQual']+X['BsmtCond']
        
        # X['HasPool'] = X['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
        # X['HasBsmt'] = X['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
        # X['HasSecondFloor'] = X['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
        
       
        
        
        # X = X.drop(['TotalBsmtSF','1stFlrSF','2ndFlrSF'],axis = 1)
        X = X.drop(['OpenPorchSF','3SsnPorch','EnclosedPorch','ScreenPorch','WoodDeckSF'],axis = 1)
        # X = X.drop(['BsmtFullBath','FullBath','BsmtHalfBath','HalfBath'],axis = 1)
        
        # X = X.drop(['OverallQual','OverallCond','ExterQual','ExterCond'],axis = 1)

        return X
    

class PolynomialFeatureCreator(BaseEstimator, TransformerMixin):
    
    def __init__(self): # no *args or **kargs
        print('Feature Pipeline!')

    def fit(self, X, y=None):
        return self # nothing else to do
        
    def transform(self, X, y=None):    
        import pandas as pd        
        X_top10 = X[['TotalBath', 'TotalSF', 'OverallQual', 'RemodSum', 
                    'GrLivArea', 'KitchenQual', 'TotRmsAbvGrd', 
                    'GarageCars', 'GarageArea','PorchSF']]
        
        
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(2)
        
        X_newfeat = poly.fit_transform(X_top10)
        
        poly_df = pd.DataFrame(X_newfeat)

        X = pd.concat([X,poly_df],axis = 1)
        
    
        return X
    


class OutlierPruner(BaseEstimator, TransformerMixin):
    
    def __init__(self,train_data): # no *args or **kargs
        # print('Dropping/modifying specific outliers')
        self.bad_indices = [524,1299,1183,692]
        self.train = train_data
        
    def fit(self, X, y=None):
        return self # nothing else to do
        
    def transform(self, X, y=None):        
        if self.train:
            X = X[~X['Id'].isin(self.bad_indices)]
            return X
        else:
            #We aren't dealing with the test set outlier
            return X
        
        
class PriceSplitter(BaseEstimator, TransformerMixin):
    
    def __init__(self,train_data): # no *args or **kargs
        # print('Separating price data and dropping indices')
        self.train = train_data
        
    def fit(self, X, y=None):
        return self # nothing else to do
        
    def transform(self, X, y=None):  
        import numpy as np
        if self.train:
            prices = np.log1p(X['SalePrice'])
            X = X.drop(['SalePrice'],axis = 1)
            X = X.drop(['Id'],axis = 1)
            return X, prices
        
        else:
            Id_df = X['Id']
            X = X.drop(['Id'],axis = 1)
            return X,Id_df
            


class FeatureSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self,train_data,corr_val,features=None): # no *args or **kargs
        # print('Selecting top features based on abs. corr_coff >',corr_val)
        self.train = train_data
        self.corr_val =corr_val
        self.feature_list= features
        
    def fit(self, X, y=None):
        return self # nothing else to do
        
    def transform(self, X, y=None):  
        import pandas as pd
        
        if self.corr_val == 0 and self.train:
            return X, X.columns.to_list()
        
        elif self.corr_val == 0 and not self.train:
            return X
        
        else:
            if self.train:
                corrmatrix = X.corr('spearman')
                cor_target = abs(corrmatrix["SalePrice"])
                #Selecting correlated features
                val = self.corr_val
                relevant_features = cor_target[cor_target>=val]
                feature_select = relevant_features.index.to_list()
                # feature_select.remove('SalePrice')
                feature_select.append('Id')
                return X[feature_select], feature_select
            
            
            
            else:
                self.feature_list.remove('SalePrice')
                return X[self.feature_list]
        
        
class DropHighCorr(BaseEstimator, TransformerMixin):
   
   def __init__(self,train_data,threshold,features=None): # no *args or **kargs
       # print('Selecting top features based on abs. corr_coff >',corr_val)
       self.train = train_data
       self.thresh = threshold
       self.feature_list= features
       
   def fit(self, X, y=None):
       return self # nothing else to do
       
   def transform(self, X, y=None):  
       import pandas as pd
       
       import numpy as np
       
       if self.train:
           feature_select = X.columns.to_list()
           corr_matrix = X.corr('pearson').abs()
           
           upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
           to_drop = [column for column in upper.columns if any(upper[column] > self.thresh)]
           X = X.drop(X[to_drop], axis=1)
           return X, X.columns.to_list()
       
       
       
       else:
           return X[self.feature_list]       
        
        
        
        
        

class FeatureTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self): # no *args or **kargs
        self.numerical_features = ['LotArea', '3SsnPorch', 'LowQualFinSF', 
                                'KitchenAbvGr', 'BsmtFinSF2', 'BsmtHalfBath', 
                                'ScreenPorch', 'EnclosedPorch', 'MasVnrArea', 
                                'OpenPorchSF', 'WoodDeckSF', 'LotFrontage', 
                                'PorchSF', 'BsmtUnfSF', '1stFlrSF', 'GrLivArea', 
                                '2ndFlrSF', 'BsmtFinSF1', 'HalfBath', 
                                'TotRmsAbvGrd', 'TotalSF', 'Fireplaces', 
                                'BsmtFullBath']
        
        
        self.further_skewed = ['3SsnPorch', 'LowQualFinSF', 'BsmtHalfBath',
                     'KitchenAbvGr', 'ScreenPorch', 'BsmtFinSF2', 
                     'EnclosedPorch', 'MasVnrArea', '2ndFlrSF', 'WoodDeckSF',
                     'HalfBath', 'OpenPorchSF', 'BsmtFullBath', 'Fireplaces']
        
        self.special_skewed =['RemodSum','GarageCars','GarageArea']
        
        self.to_drop = ['3SsnPorch','LowQualFinSF','BsmtHalfBath','ScreenPorch',
                        'BsmtFinSF2','EnclosedPorch']
        
    def fit(self, X, y=None):
        return self # nothing else to do
        
    def transform(self, X, y=None):  
        import numpy as np
        import pandas as pd
        

        from scipy.special import boxcox1p
        from scipy.stats import boxcox_normmax
        
        
        feature_select = X.columns.to_list()
        skewed_selected = list(set(self.numerical_features) & set(feature_select))
        
        further = list(set(self.further_skewed) & set(feature_select))
        special = list(set(self.special_skewed) & set(feature_select))
        
        for feature in skewed_selected:
            X[feature] = boxcox1p(X[feature], 0.70)
            
        for feature in further:
            X[feature] = np.log1p(X[feature])
            
        for feature in special:
            X[feature] = (X[feature])**2
        
        
        # X = X.drop(self.to_drop,axis = 1)
            
        return X
            
            
        



'''
import pandas as pd

#Loading training data
file_path = 'train_fix.csv'
raw_train = pd.read_csv(file_path)
FP = FeaturePruner()
CE = CategoricalTransformer()
CI = CustomImputer()
OT = OrdinalTransformer()
FC = FeatureCreator()
OP = OutlierPruner(train_data=True)
PS = PriceSplitter(train_data=True)
FS = FeatureSelector(train_data=True, corr_val = 0.3)

FT = FeatureTransformer()

X = CI.fit_transform(raw_train)
X = CE.fit_transform(X)
X = OT.fit_transform(X)
X = FP.fit_transform(X)
X = FC.fit_transform(X)
X = OP.fit_transform(X)




from scipy.stats import norm, skew
numerical_features = ['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1',
                      'BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea',
                      'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr',
                      'TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF',
                      'EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','TotalSF','PorchSF','TotalBath','RemodSum',
                     'Bedrooms/RM']


# Check the skew of all numerical features
skewed_feats = X[numerical_features].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)

skewness = skewness[abs(skewness) > 0.70]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))
print(skewness.head(30))



X,feature_select = FS.fit_transform(X)

X,prices = PS.fit_transform(X)

X= FT.fit_transform(X)




#Loading test data
file_path = 'test_fix.csv'
raw_test = pd.read_csv(file_path)
FP = FeaturePruner()
CE = CategoricalTransformer()
CI = CustomImputer()
OT = OrdinalTransformer()
FC = FeatureCreator()
OP = OutlierPruner(train_data=False)
PS = PriceSplitter(train_data=False)
FS = FeatureSelector(train_data=False, corr_val = 0.3, features=feature_select)
FT = FeatureTransformer()

X = CI.fit_transform(raw_test)
X = CE.fit_transform(X)
X = OT.fit_transform(X)
X = FP.fit_transform(X)
X = FC.fit_transform(X)
X = OP.fit_transform(X)
X = FS.fit_transform(X)


X,Id_df = PS.fit_transform(X)

X= FT.fit_transform(X)

print(X.head(10))
'''


