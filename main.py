from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import r2_score
import seaborn as sns 
from data_preparation import load_and_prepare_data, add_molecular_features
from eda import analyze_correlations, find_activity_cliffs, analyze_scaffolds
from modeling import prepare_data, prepare_model, evaluate_models, cross_validate_best_model, feature_importance_analysis
from visualization import  visualize_molecules, visualize_activity_cliffs

# File paths
data_path = 'C:/Users/tstav/Downloads/DDH Data.csv'  # Replace with your file path
properties_path = 'C:/Users/tstav/Downloads/DDH Data with Properties.csv'  # Replace with your file path

def main():
    # Data preparation
    print("Loading and preparing data...")
    data_processed = load_and_prepare_data(file_path = data_path , properties_path = properties_path )
    data_with_features = add_molecular_features(data_processed.copy())
    print(f"Data with extra info {data_with_features}")
    
    # EDA
    print("\nPerforming exploratory data analysis...")
    analyze_correlations(data_with_features.drop(columns=["SMILES", "mol", "mol_with_H"]))
    visualize_molecules(data_with_features)
    
    # Activity cliffs analysis
    print("\nFinding activity cliffs...")
    activity_cliffs = find_activity_cliffs(data_with_features)
    print(f"Found {len(activity_cliffs)} activity cliffs")
    print(f"Activity cliffs: {activity_cliffs}")
    visualize_activity_cliffs(activity_cliffs, data_with_features)
    
    # Scaffold analysis
    print("\nAnalyzing scaffolds...")
    scaffold_stats = analyze_scaffolds(data_with_features)
    #scaffold_stats.to_csv('scaffold_stats.csv', index=False)
    print(scaffold_stats)
    
    # Modeling
    print("\nBuilding and evaluating models on initial data set...")
    features, target = prepare_data(data_processed)
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2,random_state=2212)
    results_df = evaluate_models(X_train, X_test, y_train, y_test)
    
    # Cross-validation
    print("\nCross-validating best model...")
    best_model = cross_validate_best_model(features, target)
    print(f"Best model: {best_model}")
    
    # Feature importance
    print("\nAnalyzing feature importance...")
    importances = feature_importance_analysis(best_model, X_train, y_train)
    
    # Train on important features only
    important_features = importances[importances > 0].index
    X_imp_train = X_train[important_features]
    X_imp_test = X_test[important_features]
    
    model_imp = prepare_model(SVR(), X_imp_train, y_train)
    y_pred_imp = model_imp.predict(X_imp_test)
    print(f"R² score on important features: {r2_score(y_test, y_pred_imp):.4f}")
    sns.regplot(x = y_pred_imp, y = y_test)

    print("\nCross-validating best model on data set with extra features...")
    new_features, new_target = prepare_data(data_with_features.drop(columns=["mol", "mol_with_H","Scaffold"], errors='ignore'))
    new_best_model = cross_validate_best_model(new_features, new_target)
    print(f"Best model: {new_best_model}")
    
if __name__ == "__main__":
    main()