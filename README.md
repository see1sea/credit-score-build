# 🏦 Credit Score Build  
*An End-to-End Pipeline for Credit Risk Modeling with Out-of-Time Validation*

This project implements a reproducible machine learning pipeline to train an **XGBoost credit scoring model** using historical loan or application data, with rigorous **Out-of-Time (OOT) validation** to simulate real-world deployment performance and prevent temporal data leakage.

Designed for risk analysts, data scientists, and model validators, this pipeline supports regulatory-compliant model development by ensuring the test set always lies strictly in the future relative to the training data.

---

## ✨ Key Features

- **Automated Data Loading & Preprocessing**: Handles date parsing, missing value checks, and basic type alignment.
- **Flexible OOT Splitting Strategies**:
  - 📅 **Single-month holdout**: Specify `year` and `month` (e.g., `2025-12`)
  - 📆 **Custom date range**: Define `start_date` and `end_date` for flexible test windows (e.g., Q4 2025)
- **Optional Hyperparameter Tuning**:  
  Bayesian optimization via `scikit-optimize` (`skopt`) to maximize AUC on the OOT validation set.
- **Robust Model Training**:  
  Trains an XGBoost classifier with early stopping (if validation set is enabled).
- **Comprehensive Evaluation**:
  - Area Under ROC Curve (**AUC**)
  - **Accuracy**, **Precision**, **Recall**, **F1-Score**
  - Full **Classification Report** (by class)
- **Model Interpretability**:
  - Saves a feature importance plot (`results/feature_importance.png`) based on `gain`
- **Artifact Persistence**:
  - Trained model saved as `models/model.pkl` (pickle format)
  - Best hyperparameters (if tuned) → `results/best_params.json`
  - Evaluation metrics → `results/metrics.json`
  - Runtime logs → `logs/app.log`

---

## 📁 Project Structure
```bash
credit-score-build/
├── config/
│   └── config.yaml           # Main configuration file
├── data/
│   └── raw/
│       └── your_data.csv     # ← Place your dataset here
├── models/                   # Saved trained models (.pkl)
├── results/                  # Metrics, plots, reports
├── logs/                     # Execution logs
├── src/                      # Source code modules
│   ├── data_loader.py
│   ├── trainer.py
│   ├── evaluator.py
│   └── utils.py
├── main.py                   # Entry point
├── requirements.txt          # Python dependencies
├── init_dev_env.sh           # Linux/macOS setup script
├── init_dev_env.ps1          # Windows PowerShell setup script
└── README.md
```


---

## ⚙️ Quick Start Guide

### 1. Prepare Your Data
- Place your credit risk dataset as a CSV file at:  
  `data/raw/your_data.csv`
- The file must contain:
  - A **date column** indicating when each application/loan was observed (e.g., `application_date`, `observation_date`)
  - A **binary target variable** indicating default/non-default (e.g., `is_default`, `bad_flag`) — values should be `0` or `1`.

> 🔧 **Important**: Update the column names in `config/config.yaml` under the `data_schema` section:
> ```yaml
> data_schema:
>   date_col: "application_date"
>   target_col: "is_default"
> ```

### 2. Configure the Pipeline
Edit [`config/config.yaml`](config/config.yaml) to define your OOT strategy and tuning preferences:

```yaml
# OOT Split Configuration
oot_split:
  method: "single_month"      # Options: "single_month" or "date_range"
  year: 2025
  month: 12
  # start_date: "2025-10-01"  # Uncomment if using "date_range"
  # end_date: "2025-12-31"

# Hyperparameter Tuning
hyperparameter_tuning:
  enabled: true
  n_calls: 30                 # Number of Bayesian optimization iterations
  random_state: 42

# Paths
paths:
  raw_data: "data/raw/your_data.csv"
  model_output: "models/model.pkl"
  results_dir: "results/"
  logs_dir: "logs/"
```

### 3. Set Up the Environment
#### Option A: Manual Setup (All Platforms)

```shell
# Create a virtual environment
python -m venv venv

# Activate it:
#   Linux/macOS:
source venv/bin/activate
#   Windows install Ubuntu
source venv/bin/activate
# Install dependencies
pip install -r requirements.txt
```
💡 Tip: Ensure you have git and python >= 3.8 installed.
#### Option B: Use Initialization Scripts (Optional) 
If used docker, run:
Linux/macOS/Ubuntu: ./init_dev_env.sh

These scripts can auto-create directories and set permissions.

### 4. Run the Pipeline
```shell
python main.py
```
The script will:
- Load and validate your data
- Split into train (in-time) and test (OOT) sets
- (Optionally) perform hyperparameter tuning
- Train the final model
- Evaluate on the OOT test set
- Save all outputs

### 5. Review Results
After successful execution, explore:
- models/model.pkl → Loadable XGBoost model for scoring
- results/metrics.json → JSON with AUC, accuracy, etc.
- results/feature_importance.png → Visual ranking of predictive features
- logs/app.log → Timestamped log of all pipeline steps
- Example metrics.json:
```json
{
  "auc": 0.842,
  "accuracy": 0.765,
  "classification_report": {
    "0": {"precision": 0.81, "recall": 0.79, "f1-score": 0.80},
    "1": {"precision": 0.68, "recall": 0.71, "f1-score": 0.69}
  }
}
```
## 🧪 Requirements
- Python: ≥ 3.8
- Key Dependencies:
  - xgboost >= 2.0
  - scikit-learn >= 1.2
  - scikit-optimize >= 0.9 (only if tuning is enabled)
  - pandas >= 1.5
  - numpy >= 1.21
  - matplotlib >= 3.5
  - PyYAML >= 6.0

Install all via:
```shell
pip install -r requirements.txt
```
## 📄 License
This project is licensed under the MIT License — see LICENSE for details.
You are free to use, modify, and distribute this code for personal or commercial purposes.
## 🙌 Contributing
We welcome contributions! Suggested enhancements:
- Add support for LightGBM or CatBoost
- Integrate with MLflow or Weights & Biases for experiment tracking
- Add unit tests (pytest) and CI/CD (GitHub Actions)
- Support Docker containerization
- Include PSI (Population Stability Index) monitoring
To contribute:
- Fork the repository
- Create a feature branch (git checkout -b feature/your-feature)
- Commit your changes (git commit -m 'Add some feature')
- Push to the branch (git push origin feature/your-feature)
- Open a Pull Request
## 💡 Best Practice Note:
This pipeline enforces ***temporal integrity*** by design — the OOT test set always represents future observations unseen during training. This mimics real-world scoring conditions 
