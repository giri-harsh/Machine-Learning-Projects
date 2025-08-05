#%% 
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import numpy as np
# from sklearn.linear_model import LinearRegression
# %%
df = pd.read_csv("Titanic-Dataset.csv")
df.head()
# df.describe()
# %%
df.info()
df.sample(20)
# %%
df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
df.head(5)
# %%
# df.drop([""],axis=1)
df.drop(["Name","PassengerId","Embarked","SibSp","Parch" ,"Cabin" ,"Ticket"],axis=1,inplace=True)


# %%
df["Age"].fillna(df["Age"].median(), inplace=True)

df.info()

# %%

df.head()
# train test split

# %%
X = df.drop("Survived",axis=1)
y = df['Survived']
# %%
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
# %%
knn = KNeighborsClassifier(n_neighbors=4).fit(X_train,y_train)
y_pred = knn.predict(X_test)
print("Knn Result")
print(classification_report(y_test,y_pred))

# %%
# 