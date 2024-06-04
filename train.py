import pandas as pd

df = pd.read_csv('Dataset\WildBlueberry_Dataset.csv')

df.head()

df.shape

df.info()

df.isnull().sum()

"""**Checking Correlation Matrix**"""


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(17,12))
sns.set()
sns.heatmap(df.corr(), annot=True, cmap=plt.cm.CMRmap_r)
plt.show()


df.columns

df.drop(columns=['Row#','honeybee', 'MaxOfUpperTRange', 'MinOfUpperTRange', 'MaxOfLowerTRange', 'MinOfLowerTRange', 'RainingDays', 'fruitset', 'fruitmass', 'seeds'], inplace=True)

df.shape

df.columns

"""**Univariate Analysis**"""

df.plot(kind="density",
                subplots=True,
                layout = (6,3),
                figsize=(17,22),
                sharex=False,
                sharey=False);

df.plot(kind="box",
                vert=False, # makes hlots insorizontal ptead of vertical
                subplots=True,
                layout = (6,3),
                figsize=(17,22),
                sharex=False,
                sharey=False);


"""*Some of the columns have outliers. Next we will check for that!*"""

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)

df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

df.shape

"""**Training the model**"""

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

import numpy as np

X = df.drop(["yield"], axis=1)
y = df['yield']

print(X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb = XGBRegressor()
xgb.fit(X_train, y_train)
xgb_prediction = xgb.predict(X_test)

mae_xgb = mean_absolute_error(y_test, xgb_prediction)
mse_xgb = mean_squared_error(y_test, xgb_prediction)
rsq_xgb = r2_score(y_test, xgb_prediction)

print('MAE: %.3f' % mae_xgb)
print('MSE: %.3f' % mse_xgb)
print('R-Square: %.3f' % rsq_xgb)

"""**Saving the model**"""

import joblib

joblib.dump(xgb, 'xgb_model.joblib')