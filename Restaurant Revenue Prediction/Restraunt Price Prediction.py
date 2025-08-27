#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Restaurant_revenue (1).csv")
df.head()

#%%
df.info()

#%%
df.describe()

#%%
df.isnull().sum()

#%%

#%%
sns.histplot(df["Monthly_Revenue"], bins=30, kde=True)
plt.show()

#%%
sns.boxplot(x="Cuisine_Type", y="Monthly_Revenue", data=df)
plt.xticks(rotation=45)
plt.show()

#%%
sns.scatterplot(x="Average_Customer_Rating", y="Monthly_Revenue", data=df)
plt.show()

#%%
correlation = df.corr(numeric_only=True)
sns.heatmap(correlation, annot=True, cmap="coolwarm")
plt.show()

#%%
df = df.dropna()

#%%
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

encoder = OneHotEncoder(sparse_output=False, drop="first")
encoded = encoder.fit_transform(df[["Cuisine_Type"]])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(["Cuisine_Type"]))

df_model = pd.concat([df.drop(["Cuisine_Type"], axis=1).reset_index(drop=True), encoded_df], axis=1)

# Outlier treatment using IQR method
Q1 = df_model["Monthly_Revenue"].quantile(0.25)
Q3 = df_model["Monthly_Revenue"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_model = df_model[(df_model["Monthly_Revenue"] >= lower_bound) & (df_model["Monthly_Revenue"] <= upper_bound)]

#%%
X = df_model.drop("Monthly_Revenue", axis=1)
y = df_model["Monthly_Revenue"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Ridge Regression
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("R2 Score:", r2)

#%%
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Revenue")
plt.ylabel("Predicted Revenue")
plt.title("Actual vs Predicted Revenue (Ridge Regression)")
plt.show()

# %%
