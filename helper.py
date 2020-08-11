
#Helper Functions
def submission_creator(model,name):
    #Creates a submission given a fitted model    
    prediction_array = model.predict(X_submission_data)
    housing_prices =  {'Id': Id_df, 'SalePrice':np.exp(prediction_array) }
    df = pd.DataFrame(housing_prices, columns = ['Id', 'SalePrice'])
    
    submission_title = 'Submission' + name
    df.to_csv(submission_title+ '.csv',index = False)
    print('Submission Created!')


def display_scores(scores):
    #Displays score data for a model
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


def model_scorer(model):
    #Performs 10-fold cross validation for a given model and returns scores
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(model, X_train, y_train,
    scoring="neg_mean_squared_error", cv=10)
    rmse_scores = np.sqrt(-scores)
    display_scores(rmse_scores)