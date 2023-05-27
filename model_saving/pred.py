import pickle

# load the model

model = pickle.load(open("dib_79.pkl", "rb"))
output = model.predict([[1, 1, 1, 1, 1, 1, 1, 1]])
print(output)
