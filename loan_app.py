from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__, template_folder='views')

# Function to preprocess data and train model
def train_model(bank_name):
    df = pd.read_csv(f'{bank_name}.csv')
    df.drop(columns=["Applicant_ID", "Full_Name"], inplace=True)
    
    if bank_name == "BoFA":
        df.drop(columns=["Loan_Purpose"], inplace=True)
    
    label_encoders = {}
    for col in df.select_dtypes(include=["object"]).columns:
        if col != "Loan_Status":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
    
    target_encoder = LabelEncoder()
    y = df["Loan_Status"].values
    target_encoder.fit(y)
    df["Loan_Status"] = target_encoder.transform(y)
    
    X = df.drop(columns=["Loan_Status"])
    y = df["Loan_Status"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training model for {bank_name}")
    print(f"Class distribution before SMOTE: {np.bincount(y_train)}")
    
    if bank_name == "Chase":
        smote = SMOTE(random_state=42, sampling_strategy='auto')
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"Class distribution after SMOTE: {np.bincount(y_train)}")
        model = GaussianNB()
    elif bank_name in ["BoFA", "American Express"]:
        approved_count = np.sum(y_train == 1)
        rejected_count = np.sum(y_train == 0)
        scale_pos_weight = min(rejected_count / (approved_count + 1e-6), 10)  # Capped to prevent extreme bias
        print(f"Scale Pos Weight for {bank_name}: {scale_pos_weight}")
        model = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42
        )
    else:
        model = GaussianNB()
    
    model.fit(X_train, y_train)
    
    return model, scaler, label_encoders, target_encoder, list(X.columns)

# Train models for all banks
models = {}
scalers = {}
label_encoders_dict = {}
target_encoders = {}
feature_names = {}

bank_mapping = {
    "Chase": "Chase",
    "Truist": "Truist",
    "American Express": "American Express",
    "Bank of America": "BoFA"
}

for bank, bank_csv in bank_mapping.items():
    model, scaler, label_encoders, target_encoder, columns = train_model(bank_csv)
    models[bank] = model
    scalers[bank] = scaler
    label_encoders_dict[bank] = label_encoders
    target_encoders[bank] = target_encoder
    feature_names[bank] = columns

@app.route('/')
def index():
    return render_template('LoanHub.html')

@app.route('/submit', methods=['POST'])
def submit():
    user_input = request.form.to_dict()
    partner_banks = request.form.getlist('partner_banks')

    column_mapping = {
        "age": "Age",
        "annual_income": "Annual_Income",
        "bank_relationship": "Bank_Relationship_Years",
        "co_applicant": "Co_Applicant",
        "credit_score": "Credit_Score",
        "dependents": "Dependents",
        "employment_sector": "Employment_Sector",
        "employment_stability_years": "Employment_Stability_Years",
        "employment_status": "Employment_Status",
        "existing_loans": "Existing_Loans",
        "gender": "Gender",
        "home_ownership": "Home_Ownership",
        "loan_amount": "Loan_Amount",
        "loan_purpose_type": "Loan_Purpose_Type",
        "loan_term": "Loan_Term",
        "loan_type": "Loan_Type",
        "marital_status": "Marital_Status",
        "property_type": "Property_Type",
        "residency_status": "Residency_Status",
        "savings_balance": "Savings_Balance",
        "debt_to_income_ratio": "Debt_to_Income_Ratio"
    }

    data = {column_mapping.get(k, k): [v] for k, v in user_input.items() if k not in ['full_name', 'partner_banks']}
    df_input = pd.DataFrame.from_dict(data)
    print("User Input Data:")
    print(df_input)

    results = []
    for bank in partner_banks:
        model = models[bank]
        scaler = scalers[bank]
        label_encoders = label_encoders_dict[bank]
        target_encoder = target_encoders[bank]
        expected_features = feature_names[bank]
        
        print(f"Expected features for {bank}: {expected_features}")
        print(f"Input features for {bank}: {list(df_input.columns)}")

        for col, le in label_encoders.items():
            if col in df_input:
                if df_input[col][0] not in le.classes_:
                    df_input[col] = le.transform([le.classes_[0]])
                else:
                    df_input[col] = le.transform([df_input[col][0]])

        df_input = df_input.reindex(columns=expected_features, fill_value=0)
        X_scaled = scaler.transform(df_input)
        
        prediction_probabilities = model.predict_proba(X_scaled)
        prediction = model.predict(X_scaled)
        print(f"Prediction probabilities for {bank}: {prediction_probabilities}")
        print(f"Prediction for {bank}: {prediction[0]}")
        
        prediction_label = (
            target_encoder.inverse_transform([prediction[0]])[0]
            if prediction[0] in target_encoder.classes_
            else 'Unknown'
        )
        
        results.append({"bank": bank, "result": "Approved" if prediction_label == 1 else "Denied"})
    
    return render_template('result.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
