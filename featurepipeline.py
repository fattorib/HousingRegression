
from sklearn.base import BaseEstimator, TransformerMixin

class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        print('Custom imputer class. Numerical variables imputed with 0, other imputed with mode')
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X, y=None):
        missing = X.isnull().sum().sort_values(ascending=False)
        missing_features = missing.index.to_list()
        for feature in missing_features:
            if X[feature].dtypes == object or X[feature].dtypes==str :
                X[feature] = X[feature].fillna(X[feature].mode()[0])               
            else:
                X[feature] = X[feature].fillna(0)  
        return X   
        
        

class CategoricalTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self): # no *args or **kargs
        self.cat_list = ['MSZoning','Street','LotShape','LandContour','Utilities',
                        'LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType',
                        'HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType',
                        'Foundation','Heating','Electrical','CentralAir','PavedDrive','SaleType',
                        'SaleCondition']
        

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
        
        self. HouseStyle_dict=['1Story','1.5Fin','1.5Unf','2Story','2.5Fin','2.5Unf','SFoyer','SLvl']
        
        self.RoofStyle_dict=['Flat','Gable','Gambrel','Hip','Mansard','Shed']
        
        self.RoofMatl_dict=['ClyTile','CompShg','Membran','Metal','Roll','Tar&Grv','WdShake','WdShngl']
        
        self.Exterior1st_dict=['AsbShng','AsphShn','BrkComm','BrkFace','CBlock','CemntBd','HdBoard','ImStucc',
                          'MetalSd','Other','Plywood','PreCast','Stone','Stucco','VinylSd','Wd Sdng','WdShing']
        
        self.Exterior2nd_dict=['AsbShng','AsphShn','BrkComm','BrkFace','CBlock','CemntBd','HdBoard','ImStucc',
                          'MetalSd','Other','Plywood','PreCast','Stone','Stucco','VinylSd','Wd Sdng','WdShing']
        
        self. MasVnrType_dict=['BrkCmn','BrkFace','CBlock','None','Stone']
        
        self.Foundation_dict=['BrkTil','CBlock','PConc','Slab','Stone','Wood']
        
        self.Heating_dict=['Floor','GasA','GasW','Grav','OthW','Wall']
                         
        self.Electrical_dict=['SBrkr','FuseA','FuseF','FuseP','Mix']
                         
        self.CentralAir_dict = ['Y','N']
                         
        self.PavedDrive_dict = ['Y','P','N']
                         
        self.SaleType_dict=['WD','CWD','VWD','New','COD','Con','ConLw','ConLI','ConLD','Oth']
                         
        self.SaleCondition_dict=['Normal','Abnorml','AdjLand','Alloca','Family','Partial']
        
        self.all_dict = [self.MSZoning_dict,self.Street_dict,self.LotShape_dict,self.LandContour_dict,self.Utilities_dict,self.LotConfig_dict,self.LandSlope_dict,
                   self.Neighborhood_dict,self.Condition1_dict,self.Condition2_dict,self.BldgType_dict,self.HouseStyle_dict,self.RoofStyle_dict,self.RoofMatl_dict,
                   self.Exterior1st_dict,self.Exterior2nd_dict,self.MasVnrType_dict,self.Foundation_dict,self.Heating_dict,self.Electrical_dict,self.CentralAir_dict,
                  self.PavedDrive_dict,self.SaleType_dict,self.SaleCondition_dict]
        
        self.list_keys = list(zip(self.cat_list,self.all_dict))
        print('Using custom categorical transformer method')
        
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
        
        
        return X

import pandas as pd
file_path = 'train_fix.csv'
raw_df = pd.read_csv(file_path)
cat_enc = CategoricalTransformer()
CI = CustomImputer()


# print(raw_df.dtypes)



    

X = CI.fit_transform(raw_df)
X = cat_enc.fit_transform(X)
print(X.head(10))