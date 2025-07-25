
# 🚗 Smart Car Price Prediction using Machine Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

### 🎯 **Predicting Car Prices with 84% Accuracy!** 🎯

*A comprehensive machine learning project comparing basic vs. advanced preprocessing techniques for car price prediction*

[![GitHub](https://img.shields.io/badge/View_on-GitHub-black?style=flat-square&logo=github)](https://github.com/giri-harsh/car-price-prediction)
[![LinkedIn](https://img.shields.io/badge/Connect-LinkedIn-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/giri-harsh)

</div>

---

## 🌟 Project Overview

As a BTech student passionate about data science, I developed this car price prediction system during my internship to explore how different preprocessing techniques impact model performance. The project features **two distinct approaches** - a basic numerical model and an enhanced version with categorical encoding.

**What makes this project special?**
- 📈 Improved accuracy from **82% to 84%** through better preprocessing
- 🔄 Comparative analysis of basic vs. advanced feature engineering
- 🎓 Real-world application of machine learning concepts learned in academics
- 💡 Demonstrates the power of proper data preprocessing

---

## 🎯 Problem Statement

**Challenge**: Can we accurately predict a car's market price based on its features like brand, model year, fuel type, and mileage?

**Why it matters**: 
- Helps buyers make informed purchasing decisions 💰
- Assists dealers in competitive pricing strategies 📊
- Provides market insights for automotive industry analysis 🏢

---

## 📊 Dataset Overview

**Source**: [Car Price Prediction Dataset on Kaggle](https://www.kaggle.com/datasets/vijayaadithyanvg/car-price-predictionused-cars)

<details>
<summary>📋 Dataset Features (Click to expand)</summary>

| Feature | Type | Description |
|---------|------|-------------|
| **Name** | Categorical | Car brand and model |
| **Year** | Numerical | Manufacturing year |
| **Selling_Price** | Numerical | Target variable (price) |
| **Present_Price** | Numerical | Current ex-showroom price |
| **Fuel_Type** | Categorical | Petrol/Diesel/CNG |
| **Seller_Type** | Categorical | Dealer/Individual |
| **Transmission** | Categorical | Manual/Automatic |
| **Owner** | Categorical | Number of previous owners |

**Dataset Stats:**
- 📊 **Samples**: ~300 used cars
- 🏷️ **Features**: 8 input features
- 🎯 **Target**: Car selling price
- 📈 **Price Range**: ₹0.1L to ₹35L+

</details>

---

## 🛠️ Technologies & Libraries

<div align="center">

| Technology | Purpose | Version |
|------------|---------|---------|
| **Python** | Core programming language | 3.x |
| **Pandas** | Data manipulation & analysis | Latest |
| **Scikit-learn** | Machine learning algorithms | Latest |
| **LinearRegression** | Regression algorithm | - |
| **OneHotEncoder** | Categorical encoding | - |
| **train_test_split** | Data splitting | - |

</div>

---

## 🏗️ Model Architecture

### 📈 **Model 1: Basic Approach** (`car_price_prediction.py`)
```python
🔹 Features Used: Numerical features only
🔹 Algorithm: Linear Regression
🔹 Preprocessing: Minimal (basic cleaning)
🔹 Accuracy Achieved: 82% (R² Score)
```

### 🚀 **Model 2: Enhanced Approach** (`Car_Price_Prediction_PreProcessed.py`)
```python
🔹 Features Used: Numerical + Categorical (encoded)
🔹 Algorithm: Linear Regression
🔹 Preprocessing: OneHotEncoder for categorical variables
🔹 Accuracy Achieved: 84% (R² Score)
🔹 Improvement: +2% accuracy boost!
```

<details>
<summary>🧠 Why the Enhanced Model Performs Better</summary>

The enhanced model outperforms the basic version because:

1. **Captures Categorical Information**: Brand names, fuel types, and transmission types significantly impact car prices
2. **OneHot Encoding**: Converts categorical data into numerical format without imposing ordinal relationships
3. **Richer Feature Space**: More features = better pattern recognition
4. **Reduces Information Loss**: Preserves all original data characteristics

</details>

---

## 📊 Model Performance

### 🏆 **Performance Comparison**

<div align="center">

| Model | R² Score | MSE | Key Features |
|-------|----------|-----|--------------|
| **Basic Model** | **82%** | Lower | Numerical only |
| **Enhanced Model** | **84%** ⭐ | Higher | Numerical + Categorical |

</div>

### 📈 **Detailed Metrics**

<details>
<summary>📊 Performance Breakdown</summary>

**Basic Model Results:**
```
R² Score: 0.82 (82% variance explained)
MSE: [Calculated during training]
Features: Year, Present_Price, Driven_kms
```

**Enhanced Model Results:**
```
R² Score: 0.84 (84% variance explained) ⬆️
MSE: [Calculated during training]
Features: All original features (encoded)
Improvement: +2% accuracy through better preprocessing!
```

**Key Takeaway**: Even a 2% improvement is significant in real-world applications! 🎯

</details>

---

## 🚀 How to Run This Project

### 📋 Prerequisites
```bash
pip install pandas scikit-learn numpy matplotlib seaborn
```

### 🔧 Installation & Usage

1. **Clone the Repository**
   ```bash
   git clone https://github.com/giri-harsh/car-price-prediction.git
   cd car-price-prediction
   ```

2. **Run Basic Model**
   ```bash
   python car_price_prediction.py
   ```

3. **Run Enhanced Model**
   ```bash
   python Car_Price_Prediction_PreProcessed.py
   ```

4. **View Results** 📊
   Both scripts will output:
   - Model accuracy (R² score)
   - Mean Squared Error
   - Sample predictions

---

## 📂 Project Structure

```
car-price-prediction/
│
├── 📄 car_price_prediction.py           # Basic model (82% accuracy)
├── 🚀 Car_Price_Prediction_PreProcessed.py  # Enhanced model (84% accuracy)
├── 📊 dataset.csv                       # Car price dataset
├── 📝 README.md                         # Project documentation
├── 📋 requirements.txt                  # Python dependencies
├── 📸 results/                          # Output visualizations
│   ├── model_comparison.png
│   ├── feature_importance.png
│   └── prediction_analysis.png
└── 📓 notebooks/                        # Jupyter notebooks (if any)
    └── exploratory_analysis.ipynb
```

---

## 🎓 Learning Journey & Insights

<details>
<summary>💡 What I Learned from This Project</summary>

**Technical Skills Developed:**
- 🔍 **Data Preprocessing**: Importance of handling categorical variables properly
- 📊 **Feature Engineering**: How encoding techniques impact model performance
- 🎯 **Model Evaluation**: Using R² score and MSE for regression tasks
- 🔄 **Comparative Analysis**: Testing different approaches systematically

**Key Insights:**
- ✨ **Preprocessing Matters**: Even simple techniques like OneHot encoding can boost accuracy
- 📈 **Domain Knowledge**: Understanding car features helps in better feature selection
- 🎯 **Incremental Improvement**: Small improvements (2%) can have big real-world impact
- 🧠 **Learning by Doing**: Hands-on projects solidify theoretical concepts

**Challenges Overcome:**
- 🔧 Handling mixed data types (numerical + categorical)
- 📊 Choosing appropriate evaluation metrics for regression
- 🎨 Making code readable and well-documented

</details>

---

## 🔮 Future Enhancements

- [ ] 🔍 **Advanced Algorithms**: Try Random Forest, XGBoost, or Neural Networks
- [ ] 📊 **Feature Engineering**: Create new features like car age, price depreciation rate
- [ ] 🎨 **Visualization Dashboard**: Interactive plots using Plotly/Streamlit
- [ ] 🌐 **Web Application**: Deploy model using Flask/FastAPI
- [ ] 📱 **Mobile App**: Create a user-friendly mobile interface
- [ ] 🔄 **Cross-Validation**: Implement k-fold cross-validation for robust evaluation

---

## 🤝 Connect With Me

<div align="center">

**Harsh Giri** | BTech Student & Aspiring Data Scientist 🎓

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/giri-harsh)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/giri-harsh)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:2006.harshgiri@gmail.com)

*Let's connect and discuss machine learning, data science, or any exciting tech projects!*

</div>

---

## 📚 Dataset Credit

🔗 **Dataset Source**: [Car Price Prediction - Used Cars](https://www.kaggle.com/datasets/vijayaadithyanvg/car-price-predictionused-cars)

*Special thanks to the Kaggle community for providing this comprehensive dataset!*



<div align="center">

### 🌟 Found this project helpful? Give it a ⭐!

**Made with 💻 & ☕ by Harsh Giri during my internship journey**

*"Every line of code is a step closer to becoming a better data scientist!" 🚀*

---

**Contributions, suggestions, and feedback are always welcome! 🤝**

</div>

---

<sub>🚗 *"This project taught me that in machine learning, the devil is in the details - and those details make all the difference!"* - Harsh</sub>
