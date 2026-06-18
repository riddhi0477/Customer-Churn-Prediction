from flask import Flask, request, jsonify
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import shap

app = Flask(__name__)

model = joblib.load("../models/model.pkl")
scaler = joblib.load("../models/scaler.pkl")

dl_model = load_model("../models/dl_model.h5")
rnn_model = load_model("../models/rnn_model.h5")
lstm_model = load_model("../models/lstm_model.h5")

explainer = shap.KernelExplainer(model.predict, np.zeros((1, 19)))

@app.route("/predict", methods=["POST"])
def predict():
    data = np.array(request.json["features"]).reshape(1, -1)
    data_scaled = scaler.transform(data)

    ml = int(model.predict(data_scaled)[0])
    dl = float(dl_model.predict(data_scaled)[0][0])

    rnn_input = data_scaled.reshape(1,1,data_scaled.shape[1])
    rnn = float(rnn_model.predict(rnn_input)[0][0])
    lstm = float(lstm_model.predict(rnn_input)[0][0])

    # SHAP EXPLANATION
    shap_values = explainer.shap_values(data_scaled)
    importance = list(np.abs(shap_values[0]))

    return jsonify({
        "ml": ml,
        "dl": round(dl,3),
        "rnn": round(rnn,3),
        "lstm": round(lstm,3),
        "importance": importance
    })

if __name__ == "__main__":
    app.run(debug=True)