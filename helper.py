
#Helper Functions
def submission_creator(model,name):
    #Function that creates a .csv submission for test data
    import pandas as pd
    import numpy as np
    prediction_array = model.predict(X_submission_data)

    housing_prices =  {'Id': Id_df, 'SalePrice':np.exp(prediction_array) }
    df = pd.DataFrame(housing_prices, columns = ['Id', 'SalePrice'])
    
    submission_title = 'Submission' + name
    df.to_csv(submission_title+ '.csv',index = False)
    print('Submission Created!')


def display_scores(scores):
    #Function that displays scoring information about our model
    # print(scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())



def model_scorer(model,X_train, y_train):
    #Function for performing 5-fold cross-validation and scoring for model evaluation
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(model, X_train, y_train,
    scoring="neg_mean_squared_error", cv=5)
    rmse_scores = np.sqrt(-scores)
    display_scores(rmse_scores)