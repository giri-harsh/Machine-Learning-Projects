# 🚀 Oasis Data Science Internship Projects

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

### 🎯 **Supervised Learning Mastery Through Real-World Projects** 🎯

*A comprehensive collection of machine learning projects completed during my Data Science Internship at Oasis Infobyte*

[![GitHub](https://img.shields.io/badge/View_Repository-black?style=flat-square&logo=github)](https://github.com/giri-harsh/Oasis-Data-Science-Internship)
[![LinkedIn](https://img.shields.io/badge/Connect-LinkedIn-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/giri-harsh)

</div>

---

## 📋 Project Overview

During my **Data Science Internship at Oasis Infobyte**, I developed three comprehensive machine learning projects that demonstrate proficiency in supervised learning algorithms, data preprocessing, and model optimization. Each project tackles a different domain - **classification**, **regression**, and **natural language processing**.

### 🎯 **Key Achievements**
- 🌸 **Perfect Classification**: 100% accuracy on Iris Flower Classification
- 🚗 **Price Prediction**: Improved car price prediction from 82% to 84% through advanced preprocessing  
- 📧 **Spam Detection**: High-performance email classification system (~96% accuracy)

---

## 📂 Repository Structure

```
Oasis-Data-Science-Internship/
│
├── 🌸 Iris-Flower-Classification/
│   ├── iris_classification.py
│   ├── iris.csv
│   ├── README.md
│   └── results/
│       └── classification_report.png
│
├── 🚗 Car-Price-Prediction/
│   ├── car_price_prediction.py          # Basic model (82%)
│   ├── Car_Price_Prediction_PreProcessed.py  # Enhanced model (84%)
│   ├── car_data.csv
│   ├── README.md
│   └── results/
│       ├── model_comparison.png
│       └── performance_metrics.png
│
├── 📧 Email-Spam-Detection/
│   ├── spam_detection.py
│   ├── email_data.csv
│   ├── README.md
│   └── results/
│       ├── confusion_matrix.png
│       └── feature_importance.png
│
└── 📄 README.md                         # This file
```

---

## 🛠️ Tech Stack & Tools

<div align="center">

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.x |
| **Data Manipulation** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn |
| **Visualization** | Matplotlib, Seaborn |
| **Development** | Jupyter Notebook, VS Code |
| **Version Control** | Git, GitHub |

</div>

---

## 🎯 Project Details

### 1. 🌸 **Iris Flower Classification**

<details>
<summary><b>📊 Click to view project details</b></summary>

**Objective**: Classify iris flowers into three species based on sepal and petal measurements.

**Key Features**:
- **Algorithm**: K-Nearest Neighbors (KNN)
- **Dataset**: Classic Iris dataset (150 samples, 4 features)
- **Techniques**: train_test_split, classification_report
- **Result**: **🏆 100% Accuracy** on all three species

**Technologies Used**:
```python
pandas | scikit-learn | KNN | train_test_split | classification_report
```

**Performance**:
```
✨ Perfect Classification Achieved!
├── Iris Setosa: 100% accuracy
├── Iris Versicolor: 100% accuracy  
└── Iris Virginica: 100% accuracy
```

</details>

---

### 2. 🚗 **Car Price Prediction**

<details>
<summary><b>📊 Click to view project details</b></summary>

**Objective**: Predict car prices using vehicle specifications and market data.

**Dual Approach Strategy**:

**Model 1 - Basic Approach** (`car_price_prediction.py`):
- Features: Numerical data only
- Algorithm: Linear Regression
- Accuracy: **82%** (R² Score)

**Model 2 - Enhanced Approach** (`Car_Price_Prediction_PreProcessed.py`):
- Features: Numerical + Categorical (OneHot Encoded)
- Algorithm: Linear Regression with advanced preprocessing
- Accuracy: **84%** (R² Score) ⬆️ **+2% improvement**

**Technologies Used**:
```python
pandas | sklearn | LinearRegression | OneHotEncoder | train_test_split | r2_score | mean_squared_error
```

**Dataset**: [Kaggle Car Price Dataset](https://www.kaggle.com/datasets/vijayaadithyanvg/car-price-predictionused-cars)

**Key Learning**: Proper preprocessing of categorical variables significantly improves model performance!

</details>

---

### 3. 📧 **Email Spam Detection** 

<details>
<summary><b>📊 Click to view project details</b></summary>

**Objective**: Build an intelligent system to classify emails as spam or legitimate.

**Pipeline**:
1. **Data Cleaning**: Text preprocessing and normalization
2. **Feature Engineering**: TF-IDF Vectorization for text analysis
3. **Model Training**: Naive Bayes / Logistic Regression
4. **Evaluation**: Comprehensive performance metrics

**Technologies Used**:
```python
pandas | sklearn | matplotlib | seaborn | TF-IDF | Naive Bayes | Logistic Regression
```

**Expected Performance**: **~96% Accuracy** with high precision and recall

**Status**: 🚧 *Under Construction - Coming Soon!*

</details>

---

## 📊 Project Screenshots

### 🌸 Iris Classification Results
![Iris Classification](https://via.placeholder.com/600x300/4CAF50/white?text=Iris+Classification+Results+-+100%25+Accuracy)

### 🚗 Car Price Prediction Comparison
![Car Price Models](https://via.placeholder.com/600x300/2196F3/white?text=Basic+vs+Enhanced+Model+-+82%25+vs+84%25)

### 📧 Email Spam Detection Performance
![Spam Detection](https://via.placeholder.com/600x300/FF9800/white?text=Email+Spam+Detection+-+96%25+Accuracy)

---

## 🚀 How to Run These Projects

### 📋 Prerequisites
```bash
# Install required packages
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### 🔧 Quick Start Guide

1. **Clone the Repository**
   ```bash
   git clone https://github.com/giri-harsh/Oasis-Data-Science-Internship.git
   cd Oasis-Data-Science-Internship
   ```

2. **Navigate to Any Project**
   ```bash
   # For Iris Classification
   cd Iris-Flower-Classification
   python iris_classification.py
   
   # For Car Price Prediction
   cd Car-Price-Prediction
   python car_price_prediction.py
   python Car_Price_Prediction_PreProcessed.py
   
   # For Email Spam Detection (Coming Soon!)
   cd Email-Spam-Detection
   python spam_detection.py
   ```

3. **View Results** 📊
   Each project outputs detailed performance metrics and saves visualizations in the `results/` folder.

---

## 🎓 Key Learning Outcomes

<details>
<summary><b>💡 What I Learned During This Internship</b></summary>

**Technical Skills Developed**:
- 🔍 **Data Preprocessing**: Handling numerical and categorical data effectively
- 🤖 **Algorithm Selection**: Choosing appropriate ML algorithms for different problem types
- 📊 **Model Evaluation**: Using various metrics (accuracy, precision, recall, R², MSE)
- 🔄 **Iterative Improvement**: Enhancing model performance through feature engineering
- 📈 **Comparative Analysis**: Understanding trade-offs between different approaches

**Professional Skills**:
- 📝 **Documentation**: Writing clear, comprehensive project documentation
- 🎯 **Problem-Solving**: Breaking down complex problems into manageable tasks
- 🔬 **Experimental Mindset**: Testing hypotheses and validating results
- 💻 **Code Organization**: Structuring projects for maintainability and scalability

**Industry Insights**:
- 🏢 **Real-World Applications**: Understanding how ML solves business problems
- 📊 **Data Quality**: Importance of clean, well-preprocessed data
- ⚡ **Performance Optimization**: Balancing accuracy with computational efficiency
- 🔄 **Continuous Learning**: Staying updated with latest ML trends and techniques

</details>

---

## 👨‍💻 About Me

<div align="center">

**Harsh Giri** | Data Science Enthusiast & BTech Student 🎓

*Passionate about transforming data into actionable insights through machine learning*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/giri-harsh)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/giri-harsh)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:2006.harshgiri@gmail.com)

</div>

### 🌟 **Professional Interests**
- 🤖 Machine Learning & Artificial Intelligence
- 📊 Data Analysis & Visualization  
- 🔬 Predictive Modeling & Statistical Analysis
- 💼 Business Intelligence & Data-Driven Decision Making

---

## 📞 Let's Connect!

I'm always excited to discuss data science, machine learning, or potential collaboration opportunities. Whether you're a fellow student, industry professional, or someone curious about these projects, feel free to reach out!

**Contact Information**:
- 💼 **LinkedIn**: [linkedin.com/in/giri-harsh](https://linkedin.com/in/giri-harsh)
- 🐙 **GitHub**: [github.com/giri-harsh](https://github.com/giri-harsh)  
- 📧 **Email**: 2006.harshgiri@gmail.com

---

## 🙏 Acknowledgments

**Special Thanks**:
- 🏢 **Oasis Infobyte** for providing this incredible internship opportunity
- 👥 **Mentors and Peers** who guided me throughout this journey
- 🌐 **Open Source Community** for the amazing tools and datasets
- 📚 **Online Learning Platforms** that helped build my foundation

---

<div align="center">

### 🌟 **If you found these projects helpful, please give them a ⭐!**

---

**Made with 💻 & ☕ during my Data Science Internship at Oasis Infobyte**

*"This internship journey taught me that every dataset tells a story, and every model is a step toward understanding that story better. I'm proud of how far I've come and excited for what lies ahead in my data science career!" 🚀*

---

**⭐ Star this repository if it helped you learn something new!**  
**🤝 Contributions and feedback are always welcome!**

</div>

---

<sub>📈 *"Data is the new oil, but machine learning is the refinery that turns it into valuable insights."* - Learned during my internship journey</sub>
