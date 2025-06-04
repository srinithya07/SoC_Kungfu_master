import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
traindataframe = pd.read_csv('train.csv')
testdataframe = pd.read_csv('test.csv')
x = traindataframe.drop(columns=['id', 'MedHouseVal'])
y = traindataframe['MedHouseVal']
x = x.fillna(x.mean())
test_features = testdataframe.drop(columns=['id']).fillna(x.mean())
scaler = StandardScaler()
x = scaler.fit_transform(x)
test_features = scaler.transform(test_features)
x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.2, random_state=42)


class LinearRegression:
    def __init__(self, learningrate=0.001, n_iters=1000):
        self.lr = learningrate
        self.n_iters = n_iters
        self.w = None
        self.b = 0

    def fit(self, x, y):
        n_trainingset, n_features = x.shape
        self.w = np.zeros(n_features)
        self.b = 0
        for _ in range(self.n_iters):
            y_pred = np.dot(x, self.w) + self.b
            error = y_pred - y
            dw = (1/n_trainingset)*np.dot(x.T, error)
            db = (1/n_trainingset)*np.sum(error)
            self.w -= self.lr*dw
            self.b -= self.lr*db

    def predict(self, x):
        return np.dot(x, self.w) + self.b


model = LinearRegression(learningrate=0.001, n_iters=1000)
model.fit(x_train, y_train.values)
val_predictions = model.predict(x_val)
rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
print(f"Validation RMSE:{rmse}")
plt.scatter(y_val, val_predictions, alpha=0.5)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Validation:Actual vs Predicted')
plt.grid(True)
plt.show()
model.fit(x, y.values)
predictions = model.predict(test_features)
submissiondataframe = pd.DataFrame({
    'id': testdataframe['id'],
    'MedHouseVal': predictions
})
submissiondataframe.to_csv('submission.csv', index=False)
