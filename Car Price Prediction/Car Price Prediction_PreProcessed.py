#%% 
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split
# %%
df = pd.read_csv("car data.csv")
df.head()
# %%
s = (df.dtypes == "object")
cols = list(s[s].index)

# %%
ohe  = OneHotEncoder(handle_unknown='ignore',sparse_output=False)
# %%
df_fuel = pd.DataFrame(ohe.fit_transform(df[["Fuel_Type"]]))
df_fuel.columns= ohe.get_feature_names_out(["Fuel_Type"])


# %%
df.head()
# %%



# Add column names to df_fuel


df.drop('Fuel_Type',axis=1,inplace=True)

#more than one coloumn

# %%

df=pd.concat([df.reset_index(drop=True),df_fuel.reset_index(drop=True)],axis=1)
df.head()
# %%

cols.remove('Car_Name')
# cols
# %%
# cols
ohe = OneHotEncoder(handle_unknown='ignore',sparse_output=False)


# %%
df['Transmission'].unique()
df['Transmission'].value_counts()
# %%
df_transmission = pd.DataFrame(ohe.fit_transform(df[["Transmission"]]))
df_transmission.columns= ohe.get_feature_names_out(["Transmission"])
df_transmission

# %%
# now to concat the df_transmission to original df
df.drop(["Transmission"],axis=1,inplace=True)



# %%


# df["Selling_type"].unique()
# %%


df=pd.concat([df.reset_index(drop=True),df_transmission.reset_index(drop=True)],axis=1)
df.head()
# %%
df_Selling = pd.DataFrame(ohe.fit_transform(df[['Selling_type']]))
df_Selling.columns= ohe.get_feature_names_out(["Selling_type"])
df_Selling.head()
# %%

encoded_data = ohe.fit_transform(df[cols])
encoded_df = pd.DataFrame(encoded_data, columns=ohe.get_feature_names_out(cols))

df = df.drop(cols, axis=1)
df = pd.concat([df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

# %%
df.head()
# %%
df.drop(["Car_Name"],axis=1,inplace=True)
# %%


# data preprocess is now done i can make my  prediction moddel
X = df.drop(["Selling_Price"],axis=1)
X.head()
y = df["Selling_Price"]
y.head()

# %%
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# %%
model = LinearRegression().fit(X_train,y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('Mean Squared Error:', mse)
print('R-squared:', r2)


