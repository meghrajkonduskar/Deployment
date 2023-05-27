import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# load the data
df = pd.read_csv('diabetes.csv')
print(df.columns)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# train the model
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("[INFO] model trained")
score = model.score(X_test, y_test)
print(score)

# saving of model using pickle
# pickle.dump(model, open("dib_79.pkl", "wb"))

# saving of model using joblib
joblib.dump(model, "dib_79_joblib.pkl")
