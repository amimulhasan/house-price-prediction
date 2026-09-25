# 🏠 House Price Prediction using XGBoost

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-Regressor-orange)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-ML-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A machine learning project that predicts house prices using the **Boston Housing Dataset** and **XGBoost Regressor**.

> **Note:** The original `sklearn.datasets.load_boston()` has been removed in recent scikit-learn versions due to ethical concerns. This project includes a fallback that loads the dataset from the original UCI source.

---

## 📊 Project Overview

| Item | Details |
|------|---------|
| **Algorithm** | XGBoost Regressor |
| **Dataset** | Boston Housing (506 samples, 13 features) |
| **Train/Test Split** | 80% / 20% |
| **Training R² Score** | ~0.973 |
| **Test R² Score** | ~0.912 |
| **Mean Absolute Error (Test)** | ~1.99 |

### Features Used
- `CRIM` – Per capita crime rate by town  
- `ZN` – Proportion of residential land zoned for lots over 25,000 sq.ft.  
- `INDUS` – Proportion of non-retail business acres per town  
- `CHAS` – Charles River dummy variable  
- `NOX` – Nitric oxides concentration  
- `RM` – Average number of rooms per dwelling  
- `AGE` – Proportion of owner-occupied units built prior to 1940  
- `DIS` – Weighted distances to five Boston employment centres  
- `RAD` – Index of accessibility to radial highways  
- `TAX` – Full-value property-tax rate per $10,000  
- `PTRATIO` – Pupil-teacher ratio by town  
- `B` – 1000(Bk - 0.63)² where Bk is the proportion of blacks by town  
- `LSTAT` – % lower status of the population  

**Target:** `MEDV` – Median value of owner-occupied homes in $1000's

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/house-price-prediction.git
cd house-price-prediction
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the notebook
```bash
jupyter notebook Project_4_House_Price_Prediction.ipynb
```
Or open it in **VS Code**, **Google Colab**, or **JupyterLab**.

---

## 📈 Results

### Training Data
- **R² Score:** 0.9733
- **Mean Absolute Error:** 1.145

### Test Data
- **R² Score:** 0.9116
- **Mean Absolute Error:** 1.992

The model shows strong performance with high R² on both training and test sets, indicating good generalization.

---

## 📁 Project Structure

```
house-price-prediction/
│
├── Project_4_House_Price_Prediction.ipynb   # Main notebook
├── requirements.txt                         # Python dependencies
├── .gitignore                               # Git ignore rules
└── README.md                                # Project documentation
```

---

## 🛠️ Technologies Used

- **Python 3.8+**
- **NumPy** & **Pandas** – Data handling
- **Matplotlib** & **Seaborn** – Visualization
- **Scikit-learn** – Data splitting & metrics
- **XGBoost** – Gradient Boosting model

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👤 Author

Feel free to fork, star ⭐, and contribute!

---

**Made with ❤️ for learning Machine Learning**
