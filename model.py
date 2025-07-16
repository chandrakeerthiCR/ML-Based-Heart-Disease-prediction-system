import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
print(" Loading dataset...")
df = pd.read_csv('upload.csv')

# Drop unnecessary columns
if 'Id' in df.columns:
    df = df.drop('Id', axis=1)

# Define features and target
X = df.drop('target', axis=1)
y = df['target']

# Feature scaling
print(" Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Hyperparameter tuning
print("\n Tuning hyperparameters using GridSearchCV...")
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'),
                           param_grid, cv=5, scoring='accuracy', n_jobs=-1)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print(" Best Parameters Found:", grid_search.best_params_)

# Cross-validation
print("\n Performing Cross-Validation...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_model, X_scaled, y, cv=kf, scoring='accuracy')
print(f"Mean CV Accuracy: {np.mean(cv_scores) * 100:.2f}%")
print(f"CV Standard Deviation: {np.std(cv_scores) * 100:.2f}%")

# Save model and scaler
pickle.dump(best_model, open('best_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
print("\n Model saved as 'best_model.pkl'")
print(" Scaler saved as 'scaler.pkl'")

# Feature importance visualization

print("\n Plotting feature importances...")
importances = best_model.feature_importances_
features = X.columns
indices = np.argsort(importances)

plt.figure(figsize=(10,6))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.tight_layout()
plt.show()


