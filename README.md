# CSC3116 Machine Learning Project Exam - Question 3: XGBoost for MNIST Digit Recognition

This repository contains a comprehensive machine learning project implementing XGBoost classifier for MNIST digit recognition as part of the CSC3116 Machine Learning course final exam.

## Project Overview

This project implements and evaluates an XGBoost classifier for recognizing handwritten digits from the MNIST dataset. It includes:

- **Two-stage hyperparameter tuning** (RandomizedSearchCV → GridSearchCV)
- **Baseline models** (Logistic Regression, Decision Tree)
- **Ensemble models** (AdaBoost, Gradient Boosting)
- **Comprehensive evaluation** with multiple metrics
- **Visualizations** (confusion matrix, ROC curves, feature importance, learning curves)
- **Error analysis** and model comparison

## Files

- `code_source3.ipynb` - Complete Jupyter notebook with all code and analysis
- `report_question3.md` - Comprehensive written report
- `model_3.pkl` - Final trained XGBoost model (generated after running notebook)
- `mnist_train.csv` - Training dataset (not included in repo due to size)
- `mnist_test.csv` - Test dataset (not included in repo due to size)

## Requirements

- Python 3.8+
- numpy
- pandas
- scikit-learn
- xgboost
- matplotlib
- seaborn
- joblib

## Usage

1. Install dependencies:
```bash
pip install numpy pandas scikit-learn xgboost matplotlib seaborn joblib jupyter
```

2. Place `mnist_train.csv` and `mnist_test.csv` in the project directory

3. Open and run `code_source3.ipynb` in Jupyter Notebook

4. The notebook will:
   - Load and preprocess the data
   - Train baseline models
   - Perform hyperparameter tuning for XGBoost
   - Train and save the final model as `model_3.pkl`
   - Generate all visualizations
   - Perform error analysis

## Results

The final XGBoost model achieves high accuracy on the MNIST test set. See `report_question3.md` for detailed results and analysis.

## Author

CSC3116 Machine Learning Course - Question 3

## License

This project is for educational purposes.
