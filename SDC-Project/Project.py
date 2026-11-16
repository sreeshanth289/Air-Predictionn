# Generated from: Project.ipynb
# Converted at: 2026-11-15T13:42:07.001Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel("AirQualityUCI.xlsx")

df.head()

df.shape

df.info()

df.describe()

df.replace(-200,np.nan,inplace=True)

df.isnull().sum()

df.drop("NMHC(GT)", axis=1, inplace=True)

df.dropna(subset=["CO(GT)"], inplace=True)
df.drop(["Date", "Time"], axis=1, inplace=True)

df.isnull().sum()

X = df.drop("CO(GT)", axis=1)

y = df["CO(GT)"]

X.head()

y.head()

print(X.shape)
print(y.shape)

corr = df.corr()

sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.show()

corr["CO(GT)"].sort_values(ascending=False)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_train.median())

print(X_train.shape)
print(X_test.shape)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(y_test[:10].values)
print(y_pred[:10])

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

corr = X_train.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.show()

plt.figure(figsize=(7, 5))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual CO(GT)")
plt.ylabel("Predicted CO(GT)")

plt.title("Actual vs Predicted")
plt.show()

residuals = y_test - y_pred

sns.histplot(residuals, kde=True)

plt.title("Residual Distribution")
plt.show()

plt.scatter(y_pred, residuals)

plt.axhline(y=0, color="red")

plt.xlabel("Predicted Values")
plt.ylabel("Residuals")

plt.title("Residual vs Predicted")
plt.show()

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model_poly = LinearRegression()
model_poly.fit(X_train_poly, y_train)

y_pred_poly = model_poly.predict(X_test_poly)

print("MAE:", mean_absolute_error(y_test, y_pred_poly))

print("MSE:", mean_squared_error(y_test, y_pred_poly))

print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_poly)))

print("R2 Score:", r2_score(y_test, y_pred_poly))

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)

print("MAE:", mean_absolute_error(y_test, y_pred_ridge))
print("MSE:", mean_squared_error(y_test, y_pred_ridge))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_ridge)))
print("R2 Score:", r2_score(y_test, y_pred_ridge))

from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.01)
lasso.fit(X_train_scaled, y_train)
y_pred_lasso = lasso.predict(X_test_scaled)

print("MAE:", mean_absolute_error(y_test, y_pred_lasso))
print("MSE:", mean_squared_error(y_test, y_pred_lasso))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lasso)))
print("R2 Score:", r2_score(y_test, y_pred_lasso))

from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

cv = KFold(n_splits=5, shuffle=True, random_state=42)

linear = make_pipeline(
    SimpleImputer(strategy="median"),
    LinearRegression()
)

poly_model = make_pipeline(
    SimpleImputer(strategy="median"),
    PolynomialFeatures(degree=2),
    StandardScaler(),
    LinearRegression()
)

linear_scores = cross_val_score(
    linear, X, y, cv=cv, scoring="r2"
)

poly_scores = cross_val_score(
    poly_model, X, y, cv=cv, scoring="r2"
)

print("Linear CV Score:", linear_scores)
print("Linear Average:", linear_scores.mean())

print("Polynomial CV Score:", poly_scores)
print("Polynomial Average:", poly_scores.mean())

from sklearn.linear_model import Ridge

poly_ridge = make_pipeline(
    SimpleImputer(strategy="median"),
    PolynomialFeatures(degree=2),
    StandardScaler(),
    Ridge(alpha=1.0)
)

scores = cross_val_score(
    poly_ridge, X, y, cv=cv, scoring="r2"
)

print("Scores:", scores)
print("Average R2:", scores.mean())

import joblib

poly_model.fit(X_train, y_train)

saved_data = {
    "model": poly_model,
    "features": list(X.columns),
    "model_name": "Polynomial Regression"
}

joblib.dump(saved_data, "air_quality_model.pkl")