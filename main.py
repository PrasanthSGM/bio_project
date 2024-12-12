import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score
)
from time import time

# Fetch dataset
from ucimlrepo import fetch_ucirepo

# Import custom modules
from gwo import GreyWolfOptimizer
from fitness_functions import feature_selection_fitness

# Fetch and preprocess dataset
dataset = fetch_ucirepo(id=336)  # Chronic Kidney Disease Dataset
X = pd.DataFrame(dataset.data.features)
y = pd.DataFrame(dataset.data.targets)

# Preprocessing
numeric_cols = X.select_dtypes(include=["number"]).columns
categorical_cols = X.select_dtypes(exclude=["number"]).columns

# Impute missing values
imputer_numeric = SimpleImputer(strategy="mean")
X[numeric_cols] = imputer_numeric.fit_transform(X[numeric_cols])

imputer_categorical = SimpleImputer(strategy="most_frequent")
X[categorical_cols] = imputer_categorical.fit_transform(X[categorical_cols])

# Encode categorical features
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Encode target
le_y = LabelEncoder()
y = le_y.fit_transform(y.values.ravel())

# Scale numeric features
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state=42)

# Evaluation function
def evaluate_classifier(X_train, X_test, y_train, y_test, algo_name):
    clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    start_time = time()
    clf.fit(X_train, y_train)
    train_time = time() - start_time
    
    y_pred = clf.predict(X_test)
    
    return {
        'Algorithm': algo_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1 Score': f1_score(y_test, y_pred, average='weighted'),
        'Training Time (s)': train_time
    }

# 1. Baseline (All Features)
baseline_results = evaluate_classifier(X_train, X_test, y_train, y_test, 'Baseline (All Features)')

# 2. GWO Feature Selection
gwo = GreyWolfOptimizer(
    fitness_function=lambda position: feature_selection_fitness(position, X_train, y_train),
    dim=X_train.shape[1],
    pop_size=5,  # Increased population size
    max_iter=25,  # Increased iterations
    lb=0,
    ub=1,
)

start_time = time()
selected_features, best_score = gwo.optimize()
gwo_time = time() - start_time

# Get selected feature indices
selected_features_indices = [i for i in range(len(selected_features)) if selected_features[i] > 0.5]

# Prepare GWO selected features
X_train_gwo = X_train[:, selected_features_indices]
X_test_gwo = X_test[:, selected_features_indices]

# Evaluate GWO
gwo_results = evaluate_classifier(X_train_gwo, X_test_gwo, y_train, y_test, 'GWO Feature Selection')

# 3. Mutual Information Feature Selection
selector = SelectKBest(score_func=mutual_info_classif, k=len(selected_features_indices))
X_train_mi = selector.fit_transform(X_train, y_train)
X_test_mi = selector.transform(X_test)

mi_results = evaluate_classifier(X_train_mi, X_test_mi, y_train, y_test, 'Mutual Information')

# 4. Comparative Analysis Visualizations
# Prepare results
results_df = pd.DataFrame([baseline_results, gwo_results, mi_results])
results_df.to_csv('algorithm_comparison.csv', index=False)

# 1. Performance Comparison Bar Plot
plt.figure(figsize=(12, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
results_plot = results_df.set_index('Algorithm')[metrics]
results_plot.plot(kind='bar', rot=45)
plt.title('Algorithm Performance Comparison')
plt.tight_layout()
plt.savefig('performance_comparison.png')
plt.close()

# 2. GWO Convergence Visualization
plt.figure(figsize=(10, 6))
plt.plot(range(len(gwo.fitness_history)), gwo.fitness_history, marker='o')
plt.title('GWO Optimization Convergence')
plt.xlabel('Iteration')
plt.ylabel('Fitness Value')
plt.savefig('gwo_convergence.png')
plt.close()

# 3. Feature Importance Heatmap
plt.figure(figsize=(12, 8))
feature_importances = np.zeros(X.shape[1])
feature_importances[selected_features_indices] = 1
feature_importances = feature_importances.reshape(1, -1)
sns.heatmap(feature_importances, cmap='YlGnBu', xticklabels=X.columns, cbar_kws={'label': 'Selected Features'})
plt.title('GWO Selected Features')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('selected_features_heatmap.png')
plt.close()

# Print and save results
print("Algorithm Comparison:")
print(results_df)
print("\nResults saved: algorithm_comparison.csv, performance_comparison.png, "
      "gwo_convergence.png, selected_features_heatmap.png")