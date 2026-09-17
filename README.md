# Retail Demand Forecasting & Inventory Planning

An end-to-end machine learning project for forecasting retail demand and supporting inventory replenishment decisions using historical store-item sales data.

The project combines **exploratory data analysis, time-series feature engineering, machine learning, demand forecasting, and inventory planning** to convert historical sales data into actionable operational insights.

---

## 📌 Project Overview

Retail businesses need accurate demand forecasts to maintain sufficient inventory while avoiding overstocking.

This project develops a machine learning pipeline that:

* Analyzes historical sales patterns
* Identifies store and item-level demand trends
* Creates time-series and lag-based features
* Forecasts future product demand
* Evaluates forecasting performance
* Calculates safety stock requirements
* Estimates reorder points
* Generates recommended replenishment quantities
* Prioritizes inventory based on expected demand

The final output connects **machine learning predictions with practical supply-chain decisions**.

---

## 🎯 Business Problem

Poor demand forecasting can lead to:

* Stockouts and lost sales
* Excess inventory and higher holding costs
* Inefficient replenishment decisions
* Poor allocation of inventory across stores

The objective is to build a forecasting system that helps answer:

> **How much demand should we expect for each store-item combination, and when should inventory be replenished?**

---

## 📊 Dataset

The project uses historical retail sales data containing store-item-date level observations.

### Expected columns

| Column  | Description        |
| ------- | ------------------ |
| `date`  | Sales date         |
| `store` | Store identifier   |
| `item`  | Product identifier |
| `sales` | Units sold         |

The dataset should be placed in the project directory as:

```text
train.csv
```

---

## 🛠️ Technologies Used

* **Python**
* **Pandas** — data manipulation
* **NumPy** — numerical computations
* **Matplotlib** — visualization
* **Seaborn** — exploratory analysis
* **Scikit-learn** — machine learning
* **OpenPyXL** — Excel report generation

---

## 🔍 Project Workflow

### 1. Data Preparation

* Load historical sales data
* Convert date fields into datetime format
* Remove duplicate records
* Check missing values
* Validate sales values
* Sort observations chronologically

### 2. Exploratory Data Analysis

The analysis examines:

* Overall daily sales trends
* Monthly demand patterns
* Store-level sales performance
* Item-level demand
* High-demand products
* Forecasting behavior across store-item combinations

### 3. Feature Engineering

The forecasting model uses both calendar and historical demand features.

#### Calendar Features

* Year
* Month
* Week
* Day
* Day of week
* Day of year
* Quarter
* Weekend indicator

#### Cyclical Features

* Sine/cosine transformation of month
* Sine/cosine transformation of day of week

#### Lag Features

Historical demand is incorporated through:

* 1-day lag
* 7-day lag
* 14-day lag
* 28-day lag

#### Rolling Features

Rolling statistics are calculated using:

* 7-day rolling mean and standard deviation
* 14-day rolling mean and standard deviation
* 28-day rolling mean and standard deviation

These features help the model capture **short-term demand behavior, weekly seasonality, and longer-term demand patterns**.

---

## 🤖 Machine Learning Model

The project uses a:

**HistGradientBoostingRegressor**

Gradient boosting is used to model nonlinear relationships between historical demand, time-based features, and store-item characteristics.

The train-validation split is performed chronologically rather than randomly to preserve the time-series structure and avoid future-data leakage.

---

## 📈 Model Evaluation

The forecasting model is evaluated using:

### Mean Absolute Error (MAE)

Measures the average absolute difference between actual and predicted demand.

### Root Mean Squared Error (RMSE)

Penalizes larger forecasting errors more strongly.

### Mean Absolute Percentage Error (MAPE)

Measures forecasting error as a percentage of actual demand.

The notebook reports all three metrics after model training.

---

## 📦 Inventory Planning

The project extends beyond demand forecasting by translating predictions into inventory decisions.

### Safety Stock

Safety stock is estimated using forecast demand variability and lead time.

```text
Safety Stock =
Z × Demand Standard Deviation × √Lead Time
```

### Reorder Point

The reorder point is calculated as:

```text
Reorder Point =
Average Daily Demand × Lead Time
+ Safety Stock
```

### Recommended Order Quantity

A 30-day demand estimate is used to calculate an approximate replenishment quantity.

These calculations allow the forecasting model to support practical inventory management decisions.

---

## 📊 Key Outputs

The project generates:

### `forecast_results.csv`

Contains:

* Actual sales
* Predicted sales
* Forecast error
* Absolute forecast error
* Date
* Store
* Item

### `inventory_plan.csv`

Contains:

* Average daily demand
* Demand variability
* Safety stock
* Reorder point
* Recommended order quantity
* Inventory priority

### `retail_demand_analysis.xlsx`

An Excel report containing:

* Forecast results
* Inventory plan
* Inventory priority summary

---

## 📁 Project Structure

```text
Retail-Demand-Forecasting/
│
├── train.csv
│
├── retail_demand_forecasting.py
│
├── retail_demand_outputs/
│   ├── forecast_results.csv
│   ├── inventory_plan.csv
│   └── retail_demand_analysis.xlsx
│
└── README.md
```

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Retail-Demand-Forecasting
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
```

### 3. Add the dataset

Place the dataset in the project directory:

```text
train.csv
```

### 4. Run the project

```bash
python retail_demand_forecasting.py
```

The generated forecast and inventory files will be stored inside:

```text
retail_demand_outputs/
```

---

## 📌 Business Applications

This project can support retail operations through:

* Demand forecasting
* Inventory replenishment
* Safety stock planning
* Store-level inventory allocation
* Stockout prevention
* Purchase planning
* Demand variability analysis

---

## 💡 Key Skills Demonstrated

**Data Analysis**

* Data cleaning
* Exploratory data analysis
* Time-series analysis
* Aggregation and segmentation

**Machine Learning**

* Feature engineering
* Time-based validation
* Regression modeling
* Model evaluation

**Supply Chain Analytics**

* Demand forecasting
* Safety stock calculation
* Reorder point analysis
* Inventory prioritization
* Replenishment planning

**Technical Skills**

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Excel reporting

---

## 🚀 Future Improvements

Potential extensions include:

* XGBoost/LightGBM comparison
* Prophet or other dedicated forecasting approaches
* Hyperparameter optimization
* Multi-step future forecasting
* Seasonal decomposition
* Price and promotion features
* Holiday and event effects
* Economic and external demand drivers
* Automated inventory optimization
* Streamlit dashboard for interactive forecasting

---

## 👤 Author

**J. Sai Kumar**
B.Tech — Naval Architecture & Ocean Engineering
IIT Madras

---

## 📜 License

This project is intended for educational, portfolio, and analytical purposes.
