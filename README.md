# 🚀 AI & Machine Learning Projects Portfolio

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

### 🎯 **Machine Learning & AI Projects Collection** 🎯

*A comprehensive showcase of machine learning projects demonstrating various algorithms and real-world applications*

[![GitHub](https://img.shields.io/badge/View_Repository-black?style=flat-square&logo=github)](https://github.com/giri-harsh/AI-ML-Projects-Portfolio)
[![LinkedIn](https://img.shields.io/badge/Connect-LinkedIn-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/giri-harsh)

</div>

---

## 📋 Project Overview

This repository contains a diverse collection of **machine learning and AI projects** that showcase proficiency in supervised learning algorithms, data preprocessing, model optimization, and practical problem-solving. Each project tackles different domains including **classification**, **regression**, and **natural language processing**.

### 🎯 **Key Highlights**
- 🌸 **Perfect Classification**: 100% accuracy on Iris Flower Classification
- 🚗 **Price Prediction**: Advanced car price prediction with 84% accuracy through feature engineering  
- 📧 **Spam Detection**: Robust email classification system with ~96% accuracy
- 🔬 **Continuous Learning**: Regularly updated with new projects and techniques

---

## 📂 Repository Structure

```
AI-ML-Projects-Portfolio/
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
| **Machine Learning** | Scikit-learn, TensorFlow, PyTorch |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Development** | Jupyter Notebook, VS Code, Google Colab |
| **Version Control** | Git, GitHub |

</div>

---

## 🎯 Featured Projects

### 1. 🌸 **Iris Flower Classification**

<details>
<summary><b>📊 Click to view project details</b></summary>

**Objective**: Classify iris flowers into three species based on sepal and petal measurements.

**Key Features**:
- **Algorithm**: K-Nearest Neighbors (KNN)
- **Dataset**: Classic Iris dataset (150 samples, 4 features)
- **Techniques**: Cross-validation, hyperparameter tuning
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

**Objective**: Predict car prices using vehicle specifications and market data with advanced preprocessing techniques.

**Dual Model Implementation**:

**Model 1 - Baseline** (`car_price_prediction.py`):
- Features: Numerical data only
- Algorithm: Linear Regression
- Accuracy: **82%** (R² Score)

**Model 2 - Enhanced** (`Car_Price_Prediction_PreProcessed.py`):
- Features: Numerical + Categorical (OneHot Encoded)
- Algorithm: Linear Regression with feature engineering
- Accuracy: **84%** (R² Score) ⬆️ **+2% improvement**

**Technologies Used**:
```python
pandas | sklearn | LinearRegression | OneHotEncoder | feature_selection | cross_validation
```

**Dataset**: Comprehensive automotive dataset with multiple vehicle attributes

**Key Insight**: Proper preprocessing and feature engineering significantly enhance model performance!

</details>

---

### 3. 📧 **Email Spam Detection** 

<details>
<summary><b>📊 Click to view project details</b></summary>

**Objective**: Build an intelligent system to classify emails as spam or legitimate using NLP techniques.

**ML Pipeline**:
1. **Text Preprocessing**: Cleaning, tokenization, and normalization
2. **Feature Engineering**: TF-IDF Vectorization for text analysis
3. **Model Training**: Naive Bayes & Logistic Regression comparison
4. **Evaluation**: Comprehensive performance metrics and cross-validation

**Technologies Used**:
```python
pandas | sklearn | matplotlib | seaborn | TF-IDF | Naive Bayes | Logistic Regression | NLTK
```

**Performance**: **~96% Accuracy** with high precision and recall across both classes

**Advanced Features**: Feature importance analysis and confusion matrix visualization

</details>

---

## 📊 Project Visualizations

### 🌸 Iris Classification Results
![Iris Classification](https://via.placeholder.com/600x300/4CAF50/white?text=Iris+Classification+Results+-+100%25+Accuracy)

### 🚗 Car Price Prediction Comparison
![Car Price Models](https://via.placeholder.com/600x300/2196F3/white?text=Baseline+vs+Enhanced+Model+-+82%25+vs+84%25)

### 📧 Email Spam Detection Performance
![Spam Detection](https://via.placeholder.com/600x300/FF9800/white?text=Email+Spam+Detection+-+96%25+Accuracy)

---

## 🚀 Getting Started

### 📋 Prerequisites
```bash
# Install required packages
pip install pandas numpy scikit-learn matplotlib seaborn jupyter nltk
```

### 🔧 Quick Setup Guide

1. **Clone the Repository**
   ```bash
   git clone https://github.com/giri-harsh/AI-ML-Projects-Portfolio.git
   cd AI-ML-Projects-Portfolio
   ```

2. **Run Any Project**
   ```bash
   # Iris Classification
   cd Iris-Flower-Classification
   python iris_classification.py
   
   # Car Price Prediction
   cd Car-Price-Prediction
   python car_price_prediction.py
   python Car_Price_Prediction_PreProcessed.py
   
   # Email Spam Detection
   cd Email-Spam-Detection
   python spam_detection.py
   ```

3. **Explore Results** 📊
   Each project generates detailed performance metrics and saves visualizations in the `results/` folder.

---

## 🎓 Technical Skills Demonstrated

<details>
<summary><b>💡 Core Competencies Showcased</b></summary>

**Machine Learning**:
- 🔍 **Data Preprocessing**: Feature scaling, encoding, and cleaning
- 🤖 **Algorithm Implementation**: Classification, regression, and NLP models
- 📊 **Model Evaluation**: Cross-validation, metrics analysis, and performance tuning
- 🔄 **Feature Engineering**: Creating meaningful features from raw data
- 📈 **Comparative Analysis**: A/B testing different approaches and algorithms

**Programming & Tools**:
- 🐍 **Python Ecosystem**: Pandas, NumPy, Scikit-learn mastery
- 📊 **Data Visualization**: Creating insightful charts and plots
- 📝 **Documentation**: Clear code documentation and project explanations
- 🔬 **Experimental Design**: Hypothesis testing and validation
- 💻 **Code Organization**: Clean, maintainable, and scalable code structure

**Problem-Solving Approach**:
- 🎯 **End-to-End Solutions**: From data exploration to deployment-ready models
- 📊 **Business Understanding**: Translating real-world problems into ML solutions
- ⚡ **Performance Optimization**: Balancing accuracy with computational efficiency
- 🔄 **Iterative Improvement**: Continuous model refinement and enhancement

</details>

---

## 🔮 Upcoming Projects

- 🧠 **Neural Network Implementation**: Deep learning for image classification
- 🏠 **House Price Prediction**: Advanced regression with ensemble methods
- 🎭 **Sentiment Analysis**: Social media sentiment classification
- 📈 **Stock Price Forecasting**: Time series analysis and prediction
- 🖼️ **Computer Vision**: Object detection and image recognition

---

## 👨‍💻 About Me

<div align="center">

**Harsh Giri** | AI/ML Engineer & Data Science Enthusiast 🎓

*Passionate about creating intelligent solutions through machine learning and artificial intelligence*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/giri-harsh)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/giri-harsh)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:2006.harshgiri@gmail.com)

</div>

### 🌟 **Areas of Expertise**
- 🤖 Machine Learning & Deep Learning
- 📊 Data Analysis & Statistical Modeling  
- 🔬 Predictive Analytics & Feature Engineering
- 💼 End-to-End ML Pipeline Development
- 🎯 Problem-Solving with Data-Driven Approaches

---

## 📞 Let's Collaborate!

I'm always excited to discuss AI/ML projects, collaborate on interesting problems, or explore new opportunities in data science. Whether you're working on a challenging dataset, need help with model optimization, or want to brainstorm innovative solutions, let's connect!

**Get in Touch**:
- 💼 **LinkedIn**: [linkedin.com/in/giri-harsh](https://linkedin.com/in/giri-harsh)
- 🐙 **GitHub**: [github.com/giri-harsh](https://github.com/giri-harsh)  
- 📧 **Email**: 2006.harshgiri@gmail.com

**What I'm Looking For**:
- 🚀 **Open Source Contributions**: Contributing to ML libraries and tools
- 🤝 **Collaborative Projects**: Working with fellow data scientists and engineers
- 💡 **Learning Opportunities**: Exploring cutting-edge AI research and applications
- 🏢 **Professional Growth**: Full-time opportunities in AI/ML engineering

---

## 🙏 Acknowledgments

**Inspiration & Resources**:
- 🌐 **Open Source Community** for incredible tools and datasets
- 📚 **Kaggle & UCI ML Repository** for providing diverse datasets
- 👥 **Data Science Community** for knowledge sharing and support
- 🎓 **Academic Resources** that built the foundation of my ML knowledge

---

<div align="center">

### 🌟 **If you find these projects helpful, please give them a ⭐!**

---

**Built with 💻 & ☕ by a passionate ML enthusiast**
https://loan-approval-harsh-giri.streamlit.app/
https://heart-disease-prediction-harsh-giri.streamlit.app/
*"Every algorithm tells a story, every dataset holds secrets, and every model brings us closer to understanding the patterns that shape our world. This portfolio represents my journey in transforming data into intelligence!" 🚀*

---

**⭐ Star this repository if it inspired your own ML journey!**  
**🤝 Contributions, suggestions, and collaborations are always welcome!**

</div>

---

<sub>📈 *"The best way to learn machine learning is by building real projects that solve actual problems."* - Philosophy behind this portfolio</sub>
