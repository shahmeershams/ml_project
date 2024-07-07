import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
df = pd.read_csv('your_dataset.csv')  # Replace with your dataset path or URL

# Preprocess the dataset (example preprocessing)
df.fillna(0, inplace=True)
X = df.drop('target_column', axis=1)  # Replace 'target_column' with your actual target column name
y = df['target_column']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

# Save the model
joblib.dump(model, 'model.joblib')
