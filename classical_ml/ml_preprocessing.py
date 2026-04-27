# classical_ml/ml_preprocessing.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("data/breast_cancer_data.csv")

# Drop ID column if it exists
df.drop(columns=["id"], inplace=True, errors="ignore")

# Drop any empty or unnamed columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Encode diagnosis column
if 'diagnosis' in df.columns:
    le = LabelEncoder()
    df['diagnosis'] = le.fit_transform(df['diagnosis'])  # M=1, B=0

# Drop missing values
df.dropna(inplace=True)

# Save preprocessed data
df.to_csv("classical_ml/preprocessed_data.csv", index=False)

print("Preprocessing complete. Cleaned data saved.")
