from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

artifacts = joblib.load('model_artifacts.joblib')
scaler = artifacts['scaler']
knn = artifacts['knn']
le_style = artifacts['le_style']
le_label = artifacts['le_label']

@app.route('/', methods=['GET','POST'])
def index():
    recommendation = None
    if request.method == 'POST':
        try:
            time_spent = float(request.form.get('time_spent', 0))
            avg_quiz_score = float(request.form.get('avg_quiz_score', 0))
            topics_engaged = int(request.form.get('topics_engaged', 0))
            learning_style = request.form.get('learning_style', 'Visual')
            ls_enc = le_style.transform([learning_style])[0]
            X = np.array([[time_spent, avg_quiz_score, topics_engaged, ls_enc]])
            X_scaled = scaler.transform(X)
            y_pred = knn.predict(X_scaled)[0]
            recommendation = le_label.inverse_transform([y_pred])[0]
        except Exception as e:
            recommendation = f"Error: {e}"
    return render_template('index.html', recommendation=recommendation)

if __name__ == '__main__':
    app.run(debug=True)
