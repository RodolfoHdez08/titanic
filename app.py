from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Cargar modelos y codificadores
model = pickle.load(open('final_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
le_sex = pickle.load(open('le_sex.pkl', 'rb'))
le_cabin = pickle.load(open('le_cabin.pkl', 'rb'))
pca = pickle.load(open('pca.pkl', 'rb'))

@app.route('/')
def form():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Obtener datos del formulario
    pclass = int(request.form['Pclass'])
    sex = le_sex.transform([request.form['Sex']])[0]
    age = float(request.form['Age'])
    sibsp = int(request.form['SibSp'])
    parch = int(request.form['Parch'])
    fare = float(request.form['Fare'])
    cabin_letter = le_cabin.transform([request.form['Cabin_letter'].upper()])[0]
    embarked = request.form['Embarked']

    # One-hot encoding manual de 'Embarked' (drop_first=True)
    embarked_c = 1 if embarked == 'C' else 0
    embarked_q = 1 if embarked == 'Q' else 0
    # S está implícito con ambos en 0

    # Vector final de entrada
    input_data = np.array([[pclass, sex, age, sibsp, parch, fare, cabin_letter, embarked_c, embarked_q]])

    # Escalar y aplicar PCA
    input_scaled = scaler.transform(input_data)
    input_pca = pca.transform(input_scaled)

    # Predicción
    prediction = model.predict(input_pca)[0]
    result = "¡Sobrevivió! 🛟" if prediction == 1 else "No sobrevivió 💀"

    return render_template('formulario.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
