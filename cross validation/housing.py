import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

#load our data
df = pd.read_csv('housing.csv')

#split the dataset into feature and target as (x) and (y) axis

x = df[['size','bedrooms']].values
y = df['price'].values


# intiate or define our model
model =LinearRegression()

Kf= KFold(n_splits=5)

#define our crossvalidation method which is kfold
Kf= KFold(n_splits=5)

mae_score= []
for train_index,test_index in Kf.splits(x):
  X_train, X_test = x[train_index], x[test_index]
  Y_train, Y_test = y[train_index], y[test_index]
  
#training the model with set which we gets after looping
model.fit(X_train,Y_train)


#predict the test set

y_pred = model.predict(x_test)
mae = mean_absolute_error(y_test,y_pred)
mae_score.append(mae)

average_mae = np.mean(mae_score)
print(f"average mean absolute error:{average_mae}")