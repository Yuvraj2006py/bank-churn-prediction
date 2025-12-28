# Bank Churn Prediction - Data Science Project

## Project Overview

This project aims to predict bank customer churn using machine learning techniques. The dataset contains information about bank customers and whether they exited (churned) or remained with the bank.

## Dataset

- **File**: `data/Churn Modeling.csv`
- **Size**: ~10,000 customer records
- **Features**: 
  - CreditScore: Customer's credit score
  - Geography: Customer's country (France, Spain, Germany)
  - Gender: Customer's gender
  - Age: Customer's age
  - Tenure: Number of years with the bank
  - Balance: Account balance
  - NumOfProducts: Number of bank products used
  - HasCrCard: Whether customer has a credit card (1/0)
  - IsActiveMember: Whether customer is an active member (1/0)
  - EstimatedSalary: Estimated salary
- **Target Variable**: Exited (1 = churned, 0 = retained)

## Project Structure

```
Bank Churn/
├── data/                 # Dataset files
├── notebooks/            # Jupyter notebooks for EDA and modeling
├── src/                  # Source code modules
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── utils.py
├── models/               # Saved model files
├── app.py               # Streamlit web application
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Installation

1. Clone or download this repository
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Web Application

To launch the interactive Streamlit web application:

```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

### Running Jupyter Notebooks

1. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Navigate to the `notebooks/` directory
3. Open and run the notebooks in order:
   - `01_eda.ipynb`: Exploratory Data Analysis
   - `02_modeling.ipynb`: Model development and training

### Running Python Scripts

The source code modules in `src/` can be imported and used in notebooks or other scripts.

## Project Phases

1. **Phase 1**: Project Setup
2. **Phase 2**: Exploratory Data Analysis
3. **Phase 3**: Data Preprocessing & Feature Engineering
4. **Phase 4**: Model Development
5. **Phase 5**: Model Evaluation
6. **Phase 6**: Web Application Development
7. **Phase 7**: Documentation

## Technologies Used

- **Python 3.8+**
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **matplotlib & seaborn**: Data visualization
- **plotly**: Interactive visualizations
- **streamlit**: Web application framework
- **xgboost**: Gradient boosting algorithm
- **jupyter**: Interactive notebooks

## Model Performance

(To be updated after model training)

## Author

Data Science Project - Bank Churn Prediction

## License

This project is for educational purposes.

