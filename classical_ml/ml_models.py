import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define models
models = {
    "🧠 Logistic Regression": LogisticRegression(max_iter=1000),
    "🌳 Decision Tree": DecisionTreeClassifier(),
    "🌲 Random Forest": RandomForestClassifier(),
    "👟 KNN": KNeighborsClassifier(),
    "🧪 Naive Bayes": GaussianNB(),
    "⚙️ SVM": SVC(),
    "🧬 MLP Neural Net": MLPClassifier(max_iter=1000)
}

# Evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=2)
    sample_preds = y_pred[:10]
    sample_actual = y_test[:10]

    print(f"\n{name}")
    print(f"Accuracy: {acc * 100:.2f}%")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)
    print(f"Predictions: {sample_preds}")
    print(f"Actual:      {sample_actual}")
