import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import  KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

#Prepare data for classification
def prepare_data(data):
    target = data['pIC50']
    features = data.drop(['SMILES', 'pIC50'], axis=1)
    return features, target

#Pipeline
def prepare_model(algorithm, X_train, y_train): 
    model = Pipeline(steps=[
        ('preprocessing', MinMaxScaler()),
        ('algorithm', algorithm)
    ])
    model.fit(X_train, y_train)
    return model

#Regression models to be evaluated
def evaluate_models(X_train, X_test, y_train, y_test):
    algorithms = [
        RandomForestRegressor(), 
        GradientBoostingRegressor(), 
        SVR(), 
        DecisionTreeRegressor(), 
        Ridge(), 
        Lasso(),
        LinearRegression(), 
        KNeighborsRegressor(), 
        BayesianRidge()
    ]
    
    results = []
    for algorithm in algorithms:
        name = type(algorithm).__name__
        model = prepare_model(algorithm, X_train, y_train)
        pred = model.predict(X_test)
        
        results.append({
            'Algorithm': name,
            'MSE': mean_squared_error(y_test, pred),
            'MAE': mean_absolute_error(y_test, pred),
            'R2': r2_score(y_test, pred)
        })
    
    results_df = pd.DataFrame(results).sort_values(by='MSE', ascending=True)
    print(results_df)
    return results_df

#Cross-validation using the best 2 models
def cross_validate_best_model(features, target):
    def create_folds(X, k=5):
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        return [(train_idx, test_idx) for train_idx, test_idx in kf.split(X)]

    models_config = [
        {'classifier': SVR, 'hyperparameters': {'C': 1.0, 'kernel': 'linear'}},
        {'classifier': SVR, 'hyperparameters': {'C': 1.0}},
        {'classifier': SVR, 'hyperparameters': {'C': 10.0}},
        {'classifier': KNeighborsRegressor, 'hyperparameters': {'n_neighbors': 3, 'p': 1}},
        {'classifier': KNeighborsRegressor, 'hyperparameters': {'n_neighbors': 3, 'p': 2}},
        {'classifier': KNeighborsRegressor, 'hyperparameters': {'p': 1}},
        {'classifier': KNeighborsRegressor, 'hyperparameters': {'p': 2}}
    ]
    
    k_folds = create_folds(features)
    best_score = -np.inf
    best_model = None
    models_trained = 0
    
    for config in models_config:
        pipeline = Pipeline([
            ('preprocessor', MinMaxScaler()),
            ('classifier', config['classifier'](**config['hyperparameters']))
        ])
        
        fold_scores = []
        for train_idx, test_idx in k_folds:
            X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
            y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]
            
            pipeline.fit(X_train, y_train)
            models_trained += 1
            y_pred = pipeline.predict(X_test)
            fold_scores.append(r2_score(y_test, y_pred))
        
        avg_score = np.mean(fold_scores)
        if avg_score > best_score:
            best_score = avg_score
            best_model = pipeline
    
    print(f"Best Cross-Validation Score: {best_score}")
    print(f"Total Models Trained: {models_trained}")
    return best_model

#Feature selection with permutation importance
def feature_importance_analysis(model, X_train, y_train):
    result = permutation_importance(model, X_train, y_train, n_repeats=1000, random_state=42)
    importances = pd.Series(result.importances_mean, index=X_train.columns)
    
    importances.sort_values().plot(kind='barh', figsize=(10,6))
    plt.title("Permutation Importance of Features")
    plt.xlabel("Mean Importance")
    plt.show()
    
    return importances
