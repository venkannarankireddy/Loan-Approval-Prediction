import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("Loan Eligibility Prediction.csv")

# Drop ID
df.drop('Customer_ID', axis=1, inplace=True)

# Handle missing values
for col in ['Gender','Married','Dependents','Education','Self_Employed','Property_Area']:
    df[col].fillna(df[col].mode()[0], inplace=True)

for col in ['Loan_Amount','Loan_Amount_Term','Credit_History']:
    df[col].fillna(df[col].median(), inplace=True)

# Feature Engineering
df['Total_Income'] = df['Applicant_Income'] + df['Coapplicant_Income']
df.drop(['Applicant_Income','Coapplicant_Income'], axis=1, inplace=True)

# Encode categorical
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Split
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=14,
    random_state=42
)

rf.fit(X_train, y_train)

# Evaluate
print("Accuracy:", accuracy_score(y_test, rf.predict(X_test)))

# Save model (LOCAL pickle)
joblib.dump(rf, "loan_approval_model.pkl")

print("Model saved successfully.")
