from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle

app = Flask(__name__)

# Load model and metadata
model = pickle.load(open("customer_churn_model.pkl", "rb"))
feature_columns = pickle.load(open("model_features.pkl", "rb"))
encoders = pickle.load(open("label_encoders.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.form.to_dict()

    customer_name = data.get("Name", "Unknown")
    df = pd.DataFrame([data])

    # drop non-model fields
    for col in ["Name", "CustomerID"]:
        if col in df:
            df.drop(columns=[col], inplace=True)

    # encode categorical values
    for col, encoder in encoders.items():
        if col in df:
            value = df[col].iloc[0]
            if value not in encoder.classes_:
                df[col] = -1
            else:
                df[col] = encoder.transform([value])

    # add missing model columns
    for col in feature_columns:
        if col not in df:
            df[col] = 0

    # reorder
    df = df[feature_columns]

    # prediction
    prediction = model.predict(df)[0]
    churn = True if prediction == 1 else False

    return render_template(
        "result.html",
        name=customer_name,
        churn=churn,
        prediction=int(prediction)
    )

if __name__ == "__main__":
    app.run(debug=True)
