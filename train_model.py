
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

df = pd.read_csv("data.csv")

print("Dataset Shape:", df.shape)
print(df.head())

#important features

df = df[[
    "Curricular units 1st sem (approved)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 2nd sem (evaluations)",
    "Target"
]]

#handling target variable

df["Target"] = df["Target"].map({
    "Dropout": 0,
    "Enrolled": 1,
    "Graduate": 2
})

# Remove missing values
df = df.dropna()

#splittingfeatures and label

X = df.drop("Target", axis=1)
y = df["Target"]

#feature scaling 

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#training testing split 

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

#testing logistic regression
model = LogisticRegression(max_iter=300)
model.fit(X_train, y_train)

#nodel evaluation
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

#saving model
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel and scaler saved successfully!")