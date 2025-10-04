#%%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# %%
df = pd.read_csv ("Iris.csv")
df.head()
df.describe()


# %%


# task learn from meassuement of 3 diff species of iris flower and classify their types


# %%
df.drop("Id",axis = "columns",inplace= True)
df.head()
# %%


X = df[["SepalLengthCm",	"SepalWidthCm",	"PetalLengthCm"	,"PetalWidthCm"]]
y = df[["Species"]]
# %%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %%
knn= KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
# %%

y_pred = knn.predict(X)
# print(y_pred)
y_pred = knn.predict(X_test)
print("Accuracy score : ")
print(classification_report(y_test,y_pred))
