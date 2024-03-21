from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, VotingRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from xgboost import XGBRegressor
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load your machine learning models
# Note: Load the models with the same preprocessing steps you used during training

# Sample DataFrame for label encoding initialization
df_sample = pd.DataFrame(data={'Gender': ['Male'], 'Relationship Status': ['Single'],
                                'Occupation Status': ['Employed'], 'Do you use social media?': [1],
                                'What is the average time you spend on social media every day?': [2]})
a = LabelEncoder()
df_sample['Gender'] = a.fit_transform(df_sample['Gender'])
df_sample['Relationship Status'] = a.fit_transform(df_sample['Relationship Status'])
df_sample['Occupation Status'] = a.fit_transform(df_sample['Occupation Status'])
df_sample['Do you use social media?'] = a.fit_transform(df_sample['Do you use social media?'])
df_sample['What is the average time you spend on social media every day?'] = a.fit_transform(
    df_sample['What is the average time you spend on social media every day?'])

# Initialize models with the same preprocessing steps
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
catboost_classifier = CatBoostClassifier(iterations=100, learning_rate=0.1, verbose=False)
ensemble_model1 = VotingClassifier(estimators=[('Random Forest', rf_classifier), ('CatBoost', catboost_classifier)],
                                   voting='hard')

# Load your trained models
ensemble_model1.fit(df_sample, [1])  # Dummy fit, replace [1] with your target variable

catboost_model = CatBoostRegressor()
xgboost_model = XGBRegressor()
ensemble_model = VotingRegressor([('CatBoost', catboost_model), ('XGBoost', xgboost_model)])
ensemble_model.fit(df_sample, [1])  # Dummy fit, replace [1] with your target variable


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        # Sample input data, modify this according to your actual input
        user_inputs = [data.get("age"), data.get("gender"), data.get("relationship_status"),
                       data.get("occupation_status"), data.get("use_social_media"),
                       data.get("avg_time_on_social_media"), data.get("q6"), data.get("q7"), data.get("q8"),
                       data.get("q9"), data.get("q10"), data.get("q11"), data.get("q12"), data.get("q13"),
                       data.get("q14"), data.get("q15"), data.get("q16")]

        # Make predictions
        if len(df_sample) > 0 and len(df_sample.columns) > 0:
            ensemble_predictions = ensemble_model1.predict([user_inputs])
            result = {"prediction": int(ensemble_predictions[0])}
        elif len(df_sample) <= 0 and len(df_sample.columns) <= 0:
            predicted_class = catboost_classifier.predict([user_inputs])
            result = {"prediction": int(predicted_class[0])}
        else:
            result = {"prediction": 3}

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True)
