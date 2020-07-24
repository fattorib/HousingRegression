import numpy as np
import pandas as pd
pd.options.display.max_rows = 999
pd.options.display.max_columns = 400


def train_loader():
    file_path = 'train_fix.csv'
    raw_df = pd.read_csv(file_path)
    #Getting price data for training
    prices = raw_df['SalePrice']
        
    attributes_with_price = raw_df.drop(['Id'],axis = 1)
    
    attributes_with_price
    features_to_drop = ['MSSubClass','PoolQC','MiscFeature','Alley','Fence','FireplaceQu','YrSold','MoSold','MiscVal','GarageType']
    rows_to_drop = ['MasVnrArea','MasVnrType','Electrical','BsmtFinType2']
    attributes_with_price = attributes_with_price.drop(features_to_drop,axis=1)
    attributes_with_price['LotFrontage'] = attributes_with_price['LotFrontage'].fillna(0)
    
    ordinal_features = ['GarageFinish','GarageQual','GarageCond','BsmtExposure','OverallQual','OverallCond','ExterQual','ExterCond',
                        'HeatingQC','KitchenQual','Functional','BsmtFinType1','BsmtFinType2','BsmtQual','BsmtCond']
    from sklearn.preprocessing import LabelEncoder
    
    
    
    
    
    prices = attributes_with_price['SalePrice']
    attributes = attributes_with_price.drop(['SalePrice'],axis = 1)
    prices = np.log(prices)
    
    from scipy.special import boxcox1p
    lam = 0.30
    
    count_log = ['LotFrontage','LotArea','GrLivArea','TotRmsAbvGrd','1stFlrSF']

    count_box = ['BsmtUnfSF','GarageArea']

    for feature in count_log:
        attributes[feature] = np.log1p(attributes[feature])


    for feature in count_box:
        attributes[feature] = boxcox1p(attributes[feature],lam)
    
    
    #Imputing Missing values
    
    attributes['MasVnrArea'] = attributes['MasVnrArea'].fillna(0)
    attributes['MasVnrType'] = attributes['MasVnrType'].fillna('None')
    attributes['Electrical'] = attributes['Electrical'].fillna('SBrkr')
    attributes['BsmtFinType1'] = attributes['BsmtFinType1'].fillna('NA')
    attributes['BsmtFinType2'] = attributes['BsmtFinType2'].fillna('NA')
    attributes['BsmtQual'] = attributes['BsmtQual'].fillna('NA')
    attributes['BsmtCond'] =attributes['BsmtCond'].fillna('NA')
    
    
    attributes['Utilities'] =attributes['Utilities'].fillna('AllPub')
    attributes['Exterior1st'] =attributes['Exterior1st'].fillna('Other')
    attributes['Exterior2nd'] =attributes['Exterior2nd'].fillna('Other')
    attributes['BsmtExposure'] =attributes['BsmtExposure'].fillna('No')
    attributes['BsmtFinSF2'] =attributes['BsmtFinSF2'].fillna(0)
    attributes['BsmtHalfBath'] =attributes['BsmtHalfBath'].fillna(0)
    
    
    attributes['GarageYrBlt'] = attributes['GarageYrBlt'].fillna(0)
    attributes['GarageFinish'] = attributes['GarageFinish'].fillna('NA')
    attributes['GarageQual'] = attributes['GarageQual'].fillna('NA')
    attributes['GarageCond'] = attributes['GarageCond'].fillna('NA')
    attributes['SaleType'] = attributes['SaleType'].fillna('WD')
    
    # attributes_with_price = attributes_with_price.dropna(axis = 1)
    
    # print(attributes.isnull().sum())
    
    
    for feature in ordinal_features:
        #initialize new instance of labelencoder
        le = LabelEncoder()
        le.fit(attributes[feature])
        attributes[feature] = le.transform(attributes[feature])
        
        
        
    
    
    
    categorical_features = ['MSZoning','Street','LotShape','LandContour','Utilities',
                            'LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType',
                            'HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType',
                            'Foundation','Heating','Electrical','CentralAir','PavedDrive','SaleType',
                            'SaleCondition']
    
    MSZoning_dict=['A','C','FV','I','RH','RL','RP','RM','C (all)']
    
    Street_dict=['Grvl','Pave']
    
    LotShape_dict=['Reg','IR1','IR2','IR3']
    
    LandContour_dict=['Lvl','Bnk','HLS','Low']
    
    Utilities_dict=['AllPub','NoSewr','NoSeWa','ELO']
    
    LotConfig_dict=['Inside','Corner','CulDSac','FR2','FR3']
    
    LandSlope_dict=['Gtl','Mod','Sev']
    
    Neighborhood_dict=['Blmngtn','Blueste','BrDale','BrkSide','ClearCr',
                       'CollgCr','Crawfor','Edwards','Gilbert','IDOTRR',
                       'MeadowV','Mitchel','NAmes','NoRidge','NPkVill',
                       'NridgHt','NWAmes','OldTown','SWISU','Sawyer',
                       'SawyerW','Somerst','StoneBr','Timber','Veenker']
    
    Condition1_dict=['Artery','Feedr','Norm','RRNn','RRAn','PosN','PosA','RRNe','RRAe']
    
    Condition2_dict=['Artery','Feedr','Norm','RRNn','RRAn','PosN','PosA','RRNe','RRAe']
    
    BldgType_dict=['1Fam','2fmCon','Duplex','Twnhs','TwnhsE','TwnhsI']
    
    HouseStyle_dict=['1Story','1.5Fin','1.5Unf','2Story','2.5Fin','2.5Unf','SFoyer','SLvl']
    
    RoofStyle_dict=['Flat','Gable','Gambrel','Hip','Mansard','Shed']
    
    RoofMatl_dict=['ClyTile','CompShg','Membran','Metal','Roll','Tar&Grv','WdShake','WdShngl']
    
    Exterior1st_dict=['AsbShng','AsphShn','BrkComm','BrkFace','CBlock','CemntBd','HdBoard','ImStucc',
                      'MetalSd','Other','Plywood','PreCast','Stone','Stucco','VinylSd','Wd Sdng','WdShing']
    
    Exterior2nd_dict=['AsbShng','AsphShn','BrkComm','BrkFace','CBlock','CemntBd','HdBoard','ImStucc',
                      'MetalSd','Other','Plywood','PreCast','Stone','Stucco','VinylSd','Wd Sdng','WdShing']
    
    MasVnrType_dict=['BrkCmn','BrkFace','CBlock','None','Stone']
    
    Foundation_dict=['BrkTil','CBlock','PConc','Slab','Stone','Wood']
    
    Heating_dict=['Floor','GasA','GasW','Grav','OthW','Wall']
                     
    Electrical_dict=['SBrkr','FuseA','FuseF','FuseP','Mix']
                     
    CentralAir_dict = ['Y','N']
                     
    PavedDrive_dict = ['Y','P','N']
                     
    SaleType_dict=['WD','CWD','VWD','New','COD','Con','ConLw','ConLI','ConLD','Oth']
                     
    SaleCondition_dict=['Normal','Abnorml','AdjLand','Alloca','Family','Partial']
    
    all_dict = [MSZoning_dict,Street_dict,LotShape_dict,LandContour_dict,Utilities_dict,LotConfig_dict,LandSlope_dict,
               Neighborhood_dict,Condition1_dict,Condition2_dict,BldgType_dict,HouseStyle_dict,RoofStyle_dict,RoofMatl_dict,
               Exterior1st_dict,Exterior2nd_dict,MasVnrType_dict,Foundation_dict,Heating_dict,Electrical_dict,CentralAir_dict,
               PavedDrive_dict,SaleType_dict,SaleCondition_dict]
          
    
    list_keys = list(zip(categorical_features,all_dict))
    
    
    from sklearn.preprocessing import OneHotEncoder
    
    one_hot_df = pd.DataFrame(np.ones(1413),columns = ['tmp'])
    
    
    for feature,dictionary in list_keys:
        oh = OneHotEncoder()
        oh.fit(np.array(dictionary).reshape(-1, 1))
        OH_features = oh.transform(attributes[feature].values.reshape(-1, 1))
        OH_df = pd.DataFrame.sparse.from_spmatrix(OH_features,columns = dictionary)
        
        OH_df['tmp']=1
        one_hot_df['tmp']=1
        
        attributes = attributes.drop(feature,axis =1)
        one_hot_df = one_hot_df.join(OH_df,lsuffix="_left", rsuffix="_right")
        
        one_hot_df = one_hot_df.drop('tmp_right', axis=1)
        one_hot_df = one_hot_df.drop('tmp_left', axis=1)
        
        # attributes = attributes.drop('tmp', axis=1)
    
    
    
    
    
    one_hot_df['tmp']=1
    
    attributes['tmp']=1
    
    
    attributes = attributes.join(one_hot_df,how = 'left', lsuffix="_left", rsuffix="_right")
    attributes = attributes.drop('tmp_right', axis=1)
    attributes = attributes.drop('tmp_left', axis=1)
    attributes = attributes.fillna(0)
    return prices, attributes


'''
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

bestfeatures = SelectKBest(score_func=f_regression, k=100)
fit = bestfeatures.fit(attributes,prices)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(attributes.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Attributes','Score']  
print(featureScores.nlargest(100,'Score'))  
'''































