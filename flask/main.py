from flask import Flask, render_template, request
import joblib

model = joblib.load(r"..\model_saving\dib_79.pkl")

# initialise the app
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=["post"])
def predict():
    preg = int(request.form.get('preg'))
    glucose = int(request.form.get('glucose'))
    bp = int(request.form.get('bp'))
    skin = int(request.form.get('skin'))
    insulin = int(request.form.get('insulin'))
    bmi = int(request.form.get('bmi'))
    dpf = int(request.form.get('dpf'))
    age = int(request.form.get('age'))
    print(preg, glucose, bp, skin, insulin, bmi, dpf, age)
    output = model.predict([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    return "Person is diabetic" if output[0] == 1 else "Person is not diabetic"


# run the app
app.run(debug=True)
