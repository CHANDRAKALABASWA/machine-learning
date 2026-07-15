import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv('data.csv')

X = df[['time_spent','avg_quiz_score','topics_engaged','learning_style']].copy()
y = df['recommended_resource'].copy()

le_style = LabelEncoder()
X['learning_style_enc'] = le_style.fit_transform(X['learning_style'])
X = X.drop(columns=['learning_style'])

le_label = LabelEncoder()
y_enc = le_label.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X[['time_spent','avg_quiz_score','topics_engaged','learning_style_enc']],
    y_enc, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

X_test_scaled = scaler.transform(X_test)
y_pred = knn.predict(X_test_scaled)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('\nClassification Report:\n', classification_report(y_test, y_pred, target_names=le_label.classes_))

joblib.dump({'scaler': scaler, 'knn': knn, 'le_style': le_style, 'le_label': le_label}, 'model_artifacts.joblib')
print('Saved model artifacts to model_artifacts.joblib')
