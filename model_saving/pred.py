import pickle
import joblib

# load the model
# model = pickle.load(open("dib_79.pkl", "rb"))
model = joblib.load('dib_79_joblib.pkl')
output = model.predict([[1, 1, 1, 1, 1, 1, 1, 1]])
print(output)
