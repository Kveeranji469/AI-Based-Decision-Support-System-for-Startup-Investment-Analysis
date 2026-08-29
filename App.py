from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)


with open("model.pkl", "rb") as f:
    model = pickle.load(f)



@app.route("/")
def home():
    return render_template("index.html")



@app.route("/description")
def description():
    return render_template("description.html")



@app.route("/startup")
def startup():
    return render_template("predict.html")



@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_features = [
            float(request.form["funding_total_usd"]),
            float(request.form["relationships"]),
            float(request.form["age_first_funding_year"]),
            float(request.form["avg_participants"])
        ]

        final_features = np.array(input_features).reshape(1, -1)

        prediction_proba = model.predict_proba(final_features)[0][1] * 100
        prediction = round(prediction_proba, 2)

        return render_template(
            "predict.html",
            prediction_text=f"Predicted Startup Success Rate: {prediction}%"
        )

    except Exception as e:
        return render_template(
            "predict.html",
            prediction_text=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    app.run(debug=True)
