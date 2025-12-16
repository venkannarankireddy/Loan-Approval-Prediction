# 🏦 Loan Eligibility Prediction System (End-to-End ML Project)

An **end-to-end Machine Learning web application** that predicts whether a loan will be **Approved or Rejected** using a **real-world Kaggle dataset**. The project demonstrates the complete ML lifecycle — from data preprocessing and feature engineering to model training and deployment with a web-based UI.

---

## 🚀 Project Overview

Financial institutions process thousands of loan applications daily. This project shows how **Machine Learning can assist loan eligibility decisions** by learning patterns from historical applicant data.

The system:

* Takes user inputs through a web interface
* Uses a trained **Random Forest model** to predict loan eligibility
* Displays **approval probability (%)**
* Provides a **human-readable explanation** for the decision

---

## 📊 Dataset Information

* **Source:** Kaggle – Loan Eligibility Prediction Dataset
* **Rows:** 614
* **Target Variable:** `Loan_Status`
* **Type:** Binary Classification

### Features Used

* Gender
* Married
* Dependents
* Education
* Self Employed
* Loan Amount
* Loan Amount Term
* Credit History
* Property Area
* **Total Income (Engineered Feature)**

> `Customer_ID` was dropped as it is not predictive.

---

## 🧠 Machine Learning Details

### 🔹 Algorithms Explored

* Logistic Regression
* Decision Tree
* **Random Forest (Final Model)**

### 🔹 Final Model Choice

**Random Forest Classifier** was selected due to:

* Ensemble learning (better generalization)
* Robustness to noise
* Strong performance on tabular data

### 🔹 Model Performance

* **Accuracy:** ~82%

> Performance improved significantly compared to the initial small-sample version of the project.

---

## 🔧 Feature Engineering

* Created a new feature:

  ```
  Total_Income = Applicant_Income + Coapplicant_Income
  ```
* Dropped individual income columns to reduce redundancy

---

## 🛠️ Tech Stack

### Machine Learning

* Python
* pandas, numpy
* scikit-learn

### Backend

* Flask

### Frontend

* HTML
* CSS

### Tools

* Google Colab (initial experimentation)
* VS Code (local training & deployment)
* Git & GitHub

---

## 📂 Project Structure

```
Loan-Approval-Prediction/
│
├── app.py                     # Flask backend
├── train_model.py             # Local model training script
├── loan_approval_model.pkl    # Trained Random Forest model
├── Loan Eligibility Prediction.csv
│
├── templates/
│   └── index.html             # UI
│
├── static/
│   └── style.css              # Styling
│
└── README.md
```

---

## ⚙️ How to Run the Project Locally

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/Loan-Approval-Prediction.git
cd Loan-Approval-Prediction
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

Activate:

* **Windows:** `venv\\Scripts\\activate`
* **Mac/Linux:** `source venv/bin/activate`

### 3️⃣ Install Dependencies

```bash
pip install flask pandas numpy scikit-learn joblib
```

### 4️⃣ Train the Model (Local & Compatible)

```bash
python train_model.py
```

### 5️⃣ Run Flask App

```bash
python app.py
```

### 6️⃣ Open Browser

```
http://127.0.0.1:5000/
```

---

## 🧪 Application Output

* Loan Decision: **Approved / Rejected**
* Approval Probability (% confidence)
* Explanation based on income, credit history, and loan amount

---

## 📌 Important Notes

* This project is for **learning and demonstration purposes**.
* Real banking systems require additional checks, regulations, and larger datasets.
* Pickle compatibility issues were handled by **local retraining**.

---

## 📈 Future Improvements

* Add dropdowns and input validation
* Add SHAP-based explainability
* Deploy on cloud (Render / AWS / Heroku)
* Add authentication and logging

---

## 👨‍💻 Author

**Venkanna Rankireddy**
Aspiring Machine Learning Engineer

---

⭐ If you find this project useful, please consider starring the repository!
