from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def feature_selection_fitness(position, X_train, y_train):
    """
    Fitness function for feature selection using Grey Wolf Optimizer
    
    Args:
    position (np.array): Binary vector indicating selected features
    X_train (np.array): Training feature matrix
    y_train (np.array): Training target vector
    
    Returns:
    float: Fitness value (lower is better)
    """
    # Identify selected features
    selected_features = [i for i in range(len(position)) if position[i] > 0.5]

    # Penalty for no selected features
    if len(selected_features) == 0:
        return 1e10  # Very high penalty

    # Select features
    X_selected = X_train[:, selected_features]
    
    # Classifier
    clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=min(5, min(np.bincount(y_train))), shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_selected, y_train, cv=skf, scoring="accuracy")
    
    # Calculate accuracy
    accuracy = scores.mean()
    
    # Penalize number of features to encourage minimal feature set
    feature_penalty = len(selected_features) / X_train.shape[1]
    
    # Fitness function: minimize (1 - accuracy + feature_penalty)
    return 1 - accuracy + feature_penalty * 0.5