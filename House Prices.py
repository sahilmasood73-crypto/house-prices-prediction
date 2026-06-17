import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
df = pd.read_csv(url)

print("First 5 rows:")
print(df.head())
print("\nData info:")
print(df.info())
print("\nSummary stats:")
print(df.describe())

df = df.fillna(df.median(numeric_only=True))

features = ['median_income', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households']
X = df[features]
y = df['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f"\nMean Absolute Error: {mae:.2f}")
print(f"R2 Score: {r2:.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.show()

print("\nDone! Chart saved as 'actual_vs_predicted.png'")