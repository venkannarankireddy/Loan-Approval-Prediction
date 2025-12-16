print("Starting Flask app...")
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("loan_approval_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    input_features = [
    int(request.form['Gender']),
    int(request.form['Married']),
    int(request.form['Dependents']),
    int(request.form['Education']),
    int(request.form['Self_Employed']),
    int(request.form['Loan_Amount']),
    int(request.form['Loan_Amount_Term']),
    int(request.form['Credit_History']),
    int(request.form['Property_Area']),
    int(request.form['Total_Income'])
]


    # Prediction
    prediction = model.predict([input_features])[0]

    # Probability
    probability = model.predict_proba([input_features])[0][1]
    probability_percent = round(probability * 100, 2)

    if prediction == 1:
        result = "✅ Loan Approved"
        explanation = "Your financial profile looks good based on credit history and income."
    else:
        result = "❌ Loan Rejected"
        explanation = "The loan may be rejected due to low income, high loan amount, or poor credit history."

    return render_template(
        'index.html',
        prediction=result,
        probability=probability_percent,
        explanation=explanation
    )


if __name__ == "__main__":
    app.run(debug=True)
