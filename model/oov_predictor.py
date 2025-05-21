import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import xgboost as xgb
from scipy import stats

def parse_probs(df):
    # Convert string representations to actual lists
    df['parsed_probs'] = df['probs'].apply(ast.literal_eval)
    
    # Convert lists into separate features
    prob_features = pd.DataFrame(df['parsed_probs'].tolist())
    
    # Rename columns
    prob_features.columns = [f'prob_{i}' for i in range(prob_features.shape[1])]
    
    return pd.concat([df[['oov']], prob_features], axis=1)

def add_distribution_features(X):
    """Add statistical features describing the probability distribution"""
    
    # Basic descriptive statistics
    X['entropy'] = X.apply(lambda row: -np.sum(np.where(row > 0, row * np.log2(row), 0)), axis=1)
    X['variance'] = X.apply(lambda row: np.var(row), axis=1)
    X['mean'] = X.apply(lambda row: np.mean(row), axis=1)
    X['median'] = X.apply(lambda row: np.median(row), axis=1)
    X['max'] = X.apply(lambda row: np.max(row), axis=1)
    X['min'] = X.apply(lambda row: np.min(row), axis=1)
    
    # Higher-order statistics
    X['skew'] = X.apply(lambda row: stats.skew(row), axis=1)
    X['kurtosis'] = X.apply(lambda row: stats.kurtosis(row), axis=1)
    
    # Distribution shape features
    X['gini'] = X.apply(lambda row: 1 - np.sum(row**2), axis=1)  # Gini impurity
    X['top1_prob'] = X.apply(lambda row: np.max(row), axis=1)  # Highest probability
    X['top2_prob'] = X.apply(lambda row: np.partition(row, -2)[-2], axis=1)  # Second highest
    X['top1_ratio'] = X['top1_prob'] / (X['top2_prob'] + 1e-10)  # Ratio between top 1 and 2
    
    # Count-based features
    X['peaks'] = X.apply(lambda row: np.sum(row > np.mean(row)), axis=1)  # Number of above-average probs
    X['zeros'] = X.apply(lambda row: np.sum(row < 1e-10), axis=1)  # Number of near-zero probs
    X['nonzeros'] = X.apply(lambda row: np.sum(row > 1e-10), axis=1)  # Number of non-zero probs
    
    # Percentile features
    X['p90'] = X.apply(lambda row: np.percentile(row, 90), axis=1)
    X['p10'] = X.apply(lambda row: np.percentile(row, 10), axis=1)
    X['p90_p10_ratio'] = X['p90'] / (X['p10'] + 1e-10)  # Ratio between 90th and 10th percentile
    
    # Concentration metrics
    X['top5_sum'] = X.apply(lambda row: np.sum(np.partition(row, -5)[-5:]), axis=1)  # Sum of top 5 probs
    X['top5_concentration'] = X['top5_sum']  # Concentration of probability in top 5
    
    return X

def prepare_data(df):
    # Parse the probabilities
    expanded_df = parse_probs(df)
    
    # Split features and target
    X = expanded_df.drop('oov', axis=1)
    y = expanded_df['oov']
    
    # Add statistical features
    X = add_distribution_features(X)
    
    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def train_xgboost_model(X_train, X_test, y_train, y_test):
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Calculate class weights for imbalanced dataset
    n_samples = len(y_train)
    n_oov = sum(y_train)
    n_non_oov = n_samples - n_oov
    scale_pos_weight = n_non_oov / n_oov if n_oov > 0 else 1.0
    
    # Initialize XGBoost classifier with better hyperparameters
    model = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        learning_rate=0.05,        # Reduced learning rate for better generalization
        n_estimators=200,          # More trees
        max_depth=4,               # Slightly deeper trees
        min_child_weight=2,        # Helps prevent overfitting
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=1,                   # Minimum loss reduction for split
        reg_alpha=0.1,             # L1 regularization
        reg_lambda=1,              # L2 regularization
        eval_metric='auc',
        objective='binary:logistic',
        random_state=42
    )
    
    # Train the model with early stopping
    model.fit(
        X_train_scaled, 
        y_train,
        eval_set=[(X_test_scaled, y_test)],
        # early_stopping_rounds=20,  # Re-enabled with higher patience
        verbose=False
    )
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:,1]
    
    # Evaluate
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("AUC-ROC:", roc_auc_score(y_test, y_prob))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importance
    })
    
    top_features = feature_importance.sort_values('Importance', ascending=False)
    print("\nTop 15 Most Important Features:")
    print(top_features.head(15))
    
    return model, scaler, top_features

def save_model(model, scaler):
    # Save the model and scaler
    model.save_model('train_oov_predictor.json')
    import joblib
    joblib.dump(scaler, 'train_oov_scaler.pkl')

def load_model():
    # Load the model and scaler
    import joblib
    model = xgb.XGBClassifier()
    model.load_model('train_oov_predictor.json')
    scaler = joblib.load('train_oov_scaler.pkl')
    return model, scaler

def main_xgboost():
    df = pd.read_csv('train_oov_preds.csv')
    X_train, X_test, y_train, y_test = prepare_data(df)
    model, scaler, top_features = train_xgboost_model(X_train, X_test, y_train, y_test)

    # Save the model and scaler
    model.save_model('train_oov_predictor.json')
    import joblib
    joblib.dump(scaler, 'train_oov_scaler.pkl')

    return model, scaler, X_train, X_test, y_train, y_test, top_features
