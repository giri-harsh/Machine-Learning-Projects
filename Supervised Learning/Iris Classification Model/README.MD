# 🌸 Iris Flower Classification using K-Nearest Neighbors

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

### 🎯 **Perfect Classification Achieved: 100% Accuracy!** 🎯

*A machine learning model that classifies iris flowers with perfect precision using the K-Nearest Neighbors algorithm*

</div>

---

## 🌺 About The Project

This project implements a **K-Nearest Neighbors (KNN) classifier** to identify iris flower species based on their physical characteristics. Built as an internship assignment, this model demonstrates the power of supervised learning by achieving **100% accuracy** across all three iris species!

### 🔍 Problem Statement
Given the sepal and petal measurements of an iris flower, can we accurately predict which of the three species it belongs to?
- **Iris Setosa** 🌸
- **Iris Versicolor** 🌺  
- **Iris Virginica** 🌼

### 📊 About the Dataset
The famous [Iris dataset](https://www.kaggle.com/datasets/saurabh00007/iriscsv/code) contains 150 samples of iris flowers with:
- **4 Features**: Sepal Length, Sepal Width, Petal Length, Petal Width
- **3 Classes**: Setosa, Versicolor, Virginica
- **50 samples** per species

---

## 🛠️ Technologies Used

<details>
<summary>Click to expand tech stack</summary>

| Technology | Purpose |
|------------|---------|
| **Python 3.x** | Core programming language |
| **Pandas** | Data manipulation and analysis |
| **Scikit-learn** | Machine learning algorithms |
| **KNN Classifier** | Classification algorithm |
| **train_test_split** | Data splitting for training/testing |
| **classification_report** | Model performance evaluation |

</details>

---

## 🚀 Quick Start Guide

### Prerequisites
```bash
pip install pandas scikit-learn numpy
```

### Installation & Usage
1. **Clone the repository**
   ```bash
   git clone https://github.com/giri-harsh/iris-flower-classification.git
   cd iris-flower-classification
   ```

2. **Run the model**
   ```bash
   python iris_classification.py
   ```

3. **View results** 📈
   The script will output the classification report and accuracy metrics!

---

## 📈 Model Performance

### 🎉 **PERFECT SCORE ACHIEVED!**

```
🏆 CLASSIFICATION RESULTS 🏆
================================
Overall Accuracy: 100.00% ✨

Species-wise Performance:
┌─────────────────┬───────────┬────────┬─────────┬─────────┐
│     Species     │ Precision │ Recall │ F1-Score│ Support │
├─────────────────┼───────────┼────────┼─────────┼─────────┤
│ Iris-setosa     │   1.00    │  1.00  │   1.00  │   XX    │
│ Iris-versicolor │   1.00    │  1.00  │   1.00  │   XX    │  
│ Iris-virginica  │   1.00    │  1.00  │   1.00  │   XX    │
└─────────────────┴───────────┴────────┴─────────┴─────────┘

🎯 Perfect Classification - No Misclassifications!
```

<details>
<summary>📊 Confusion Matrix Visualization</summary>

```
Confusion Matrix:
     Predicted
      Set Ver Vir
Set [[ X   0   0]]  ← Perfect Setosa Classification
Ver [[ 0   X   0]]  ← Perfect Versicolor Classification  
Vir [[ 0   0   X]]  ← Perfect Virginica Classification
     ↑
   Actual
```

*Where X represents the perfect predictions for each species!*

</details>

---

## 📂 Project Structure

```
iris-flower-classification/
│
├── 📄 iris_classification.py    # Main classification script
├── 📊 iris.csv                  # Dataset file
├── 📝 README.md                 # Project documentation
├── 📋 requirements.txt          # Dependencies
└── 📸 results/                  # Output visualizations
    ├── confusion_matrix.png
    └── classification_report.txt
```

---

## 🧠 How It Works

<details>
<summary>🔍 Algorithm Breakdown</summary>

1. **Data Loading**: Import iris dataset using Pandas
2. **Data Preprocessing**: Clean and prepare features
3. **Train-Test Split**: Divide data (typically 80-20 split)
4. **KNN Training**: Train K-Nearest Neighbors classifier
5. **Prediction**: Classify test samples
6. **Evaluation**: Generate comprehensive performance metrics

**Why KNN Works So Well Here:**
- Iris species have distinct feature patterns
- Small, clean dataset with no noise
- Clear decision boundaries between classes
- KNN's distance-based approach perfectly captures the natural clustering

</details>

---

## 🎯 Key Insights

- 🌟 **100% Accuracy** demonstrates the high quality of the iris dataset
- 🔍 KNN algorithm is perfect for this type of classification problem
- 📊 All three species have distinct, separable characteristics
- 🚀 This project showcases fundamental ML concepts beautifully

---

## 🤝 Connect With Me

<div align="center">

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/giri-harsh)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/giri-harsh)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:2006.harshgiri@gmail.com)

</div>

---

## 📚 Dataset Source

🔗 **Original Dataset**: [Iris CSV on Kaggle](https://www.kaggle.com/datasets/saurabh00007/iriscsv/code)

---

## 🎪 What's Next?

- [ ] Add data visualization plots
- [ ] Implement other algorithms (SVM, Random Forest)
- [ ] Create a web interface for predictions
- [ ] Add cross-validation analysis

---

<div align="center">

### 🌟 If you found this project helpful, please give it a ⭐!

**Made with 💻 & ☕ by Harsh Giri**

*Contributions, issues, and feature requests are welcome! 🚀*

</div>

---

<sub>🌸 *"In every dataset, there's a story waiting to be discovered. The Iris dataset taught me that sometimes, perfection is achievable!"* - Harsh</sub>
