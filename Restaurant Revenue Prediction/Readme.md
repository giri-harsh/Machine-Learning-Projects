# Restaurant Monthly Revenue Prediction

This project predicts **monthly revenue of restaurants** using a dataset from Kaggle:  
[Restaurants Revenue Prediction Dataset](https://www.kaggle.com/datasets/mrsimple07/restaurants-revenue-prediction).

## Project Overview
The dataset contains information about restaurants, including customer numbers, menu pricing, marketing spend, cuisine type, promotions, reviews, and customer spending patterns.  
We aim to predict **monthly revenue** using a Machine Learning model.

## Steps in the Project

### 1. Exploratory Data Analysis (EDA)
- Checked dataset shape, datatypes, and missing values
- Visualized distributions of numerical features
- Compared revenue across different cuisine types
- Analyzed relationships between features and monthly revenue

### 2. Data Preprocessing
- Removed missing values (if any)
- Handled categorical feature `Cuisine_Type` using **One-Hot Encoding**
- Removed outliers from `Monthly_Revenue` using the **IQR method**
- Scaled numerical features using **StandardScaler**

### 3. Data Visualization
- Histograms of revenue distribution
- Boxplots for revenue by cuisine type
- Correlation heatmap of numerical variables
- Scatter plots for revenue vs. key features

### 4. Model Building
- Used **Ridge Regression** as the prediction model
- Split dataset into training (80%) and testing (20%)
- Evaluated with **Mean Squared Error (MSE)** and **R² Score**

### 5. Results
- Achieved MSE ≈ *value depends on run*
- R² Score ≈ *value depends on run*
- Model gives reasonable intermediate-level performance for revenue prediction

## Tech Stack
- Python
- Pandas, NumPy
- Matplotlib, Seaborn (visualization)
- Scikit-learn (preprocessing, modeling)

## How to Run
1. Clone the repository or download the dataset from Kaggle.
2. Place the dataset CSV file in your working directory.
3. Run the Python script or Jupyter Notebook sections marked with `#%%`.

## Future Improvements
- Hyperparameter tuning of Ridge Regression
- Try Lasso or ElasticNet for feature selection
- Deploy the model as a simple web app (e.g., Streamlit)
