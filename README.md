# Multi-Output Regression: Energy Efficiency Dataset

## Question 2 - Machine Learning Exam

This repository contains a comprehensive multi-output regression pipeline for predicting heating load and cooling load of buildings based on eight building features.

## Project Overview

The project implements and compares multiple regression models to predict:
- **Heating Load**: Energy required to heat a building
- **Cooling Load**: Energy required to cool a building

## Dataset

The energy efficiency dataset contains:
- **8 Input Features**: Relative Compactness, Surface Area, Wall Area, Roof Area, Overall Height, Orientation, Glazing Area, Glazing Area Distribution
- **2 Target Variables**: Heating Load, Cooling Load
- **Training Set**: 614 samples
- **Test Set**: 154 samples

## Files Structure

```
├── code_source2.ipynb          # Jupyter notebook with complete analysis
├── code_source2.py              # Python script version
├── report_question2.md         # Comprehensive analysis report
├── energy efficiency dataset_train.csv
├── energy efficiency dataset_test.csv
├── energy efficiency dataset.csv
└── README.md                    # This file
```

## Models Implemented

1. **Linear Regression** (Baseline)
2. **Polynomial Regression** (degree 2-3, tuned via CV)
3. **Ridge Regression** (alpha tuned via grid search)
4. **Lasso Regression** (alpha tuned via grid search)
5. **Support Vector Regression (SVR)** (RBF kernel, hyperparameters tuned)

## Features

- ✅ 5-fold cross-validation for all models
- ✅ Comprehensive evaluation metrics (MSE, MAE, RMSE, MAPE, R²)
- ✅ Feature importance analysis (correlations and coefficients)
- ✅ Model comparison visualizations
- ✅ Learning curves for selected models
- ✅ Model persistence (saved as .pkl files)
- ✅ Complete documentation and report

## Usage

### Running the Notebook

1. Open `code_source2.ipynb` in Jupyter Notebook or JupyterLab
2. Run all cells sequentially
3. Generated outputs:
   - `model1_2.pkl` - Heating load predictor
   - `model2_2.pkl` - Cooling load predictor
   - `figures/` - All visualizations
   - `*.csv` - Results tables

### Running the Python Script

```bash
python code_source2.py
```

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
scipy
joblib
jupyter (for notebook)
```

## Results

All models are evaluated using:
- Cross-validation metrics (mean ± std)
- Test set performance metrics
- Feature importance analysis
- Comprehensive visualizations

## Report

See `report_question2.md` for:
- Detailed methodology
- Model comparison results
- Feature importance insights
- Hyperparameter analysis
- Conclusions and recommendations

## Author

Machine Learning Exam - Question 2

## License

This project is for educational purposes.

