import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

print("Loading dataset...")
df = pd.read_excel("customer_churn_large_dataset.xlsx")

# ----- Step 1: Generate synthetic churn -----

print("Generating synthetic churn column...")

# Normalize numeric features
df["bill_norm"] = df["Monthly_Bill"] / df["Monthly_Bill"].max()
df["usage_norm"] = 1 - (df["Total_Usage_GB"] / df["Total_Usage_GB"].max())
df["sub_norm"] = 1 - (df["Subscription_Length_Months"] / df["Subscription_Length_Months"].max())
df["age_norm"] = df["Age"] / df["Age"].max()

# Weighted churn probability
df["churn_prob"] = (
    0.40 * df["bill_norm"] +
    0.30 * df["usage_norm"] +
    0.20 * df["sub_norm"] +
    0.10 * df["age_norm"]
)

# Convert probability to churn label
df["Churn"] = (df["churn_prob"] > df["churn_prob"].median()).astype(int)

# Drop helper columns
df = df.drop(columns=["bill_norm", "usage_norm", "sub_norm", "age_norm", "churn_prob"])

print("Synthetic churn column added successfully!")

# ----- Step 2: Encode categorical fields -----

print("Encoding categorical columns...")
encoders = {}
categorical_cols = ["Gender", "Location"]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ----- Step 3: Select features -----

feature_columns = [
    "Age",
    "Gender",
    "Location",
    "Subscription_Length_Months",
    "Monthly_Bill",
    "Total_Usage_GB"
]

X = df[feature_columns]
y = df["Churn"]

# ----- Step 4: Train model -----

print("Training model...")
model = RandomForestClassifier()
model.fit(X, y)

# ----- Step 5: Save files -----

print("Saving model and encoders...")

pickle.dump(model, open("customer_churn_model.pkl", "wb"))
pickle.dump(feature_columns, open("model_features.pkl", "wb"))
pickle.dump(encoders, open("label_encoders.pkl", "wb"))

print("\n====================================")
print("Model training completed successfully!")
print("Files generated:")
print("- customer_churn_model.pkl")
print("- model_features.pkl")
print("- label_encoders.pkl")
print("====================================")
