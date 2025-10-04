#Predict Selling Price of the car


#%%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,r2_score

# %%
df = pd.read_csv("car data.csv")
df.head()
df.describe()
# %%


X = df.drop(['Car_Name','Selling_Price','Fuel_Type',	'Selling_type'	,'Transmission'],axis=1)
y = df['Selling_Price']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# %%
# X.head()
# y.head()



# %%
#training the model

model = LinearRegression().fit(X_train,y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('Mean Squared Error:', mse)
print('R-squared:', r2)


# %%


#testing our own prediction
df

# %%
