# %% [markdown]
# RETAIL DEMAND FORECASTING & INVENTORY PLANNING
#
# End-to-end machine learning project covering:
# - Exploratory Data Analysis
# - Time-series feature engineering
# - Demand forecasting
# - Model evaluation
# - Forecast error analysis
# - Safety stock calculation
# - Reorder point calculation
# - Inventory prioritization
# - Business insights
#
# Dataset expected:
# train.csv
#
# Required columns:
# date, store, item, sales


# %%
# IMPORT LIBRARIES

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)


# %%
# CONFIGURATION

DATA_PATH = "train.csv"
OUTPUT_DIR = Path("retail_demand_outputs")

OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


# %%
# LOAD DATASET

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst five rows:")
print(df.head())


# %%
# DATA INSPECTION

print("DATA TYPES")
print(df.dtypes)

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDUPLICATE ROWS")
print(df.duplicated().sum())

print("\nDATASET INFORMATION")
print(f"Rows   : {len(df):,}")
print(f"Columns: {len(df.columns)}")


# %%
# DATA CLEANING

df["date"] = pd.to_datetime(df["date"])

df = df.drop_duplicates()

df = df.sort_values(
    ["store", "item", "date"]
).reset_index(drop=True)

negative_sales = (df["sales"] < 0).sum()

print("Negative sales records:", negative_sales)

if negative_sales > 0:
    df = df[df["sales"] >= 0].copy()

print("\nCleaned dataset shape:", df.shape)
print("Date range:", df["date"].min(), "to", df["date"].max())
print("Stores:", df["store"].nunique())
print("Items:", df["item"].nunique())


# %%
# DAILY SALES TREND

daily_sales = (
    df.groupby("date")["sales"]
      .sum()
)

plt.figure(figsize=(14, 5))

plt.plot(
    daily_sales.index,
    daily_sales.values
)

plt.title("Daily Sales Trend")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# %%
# MONTHLY SALES TREND

monthly_sales = (
    df.set_index("date")
      .resample("ME")["sales"]
      .sum()
)

plt.figure(figsize=(14, 5))

plt.plot(
    monthly_sales.index,
    monthly_sales.values,
    marker="o"
)

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")

plt.xticks(rotation=45)

plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# %%
# STORE PERFORMANCE

store_sales = (
    df.groupby("store")["sales"]
      .sum()
      .sort_values(ascending=False)
)

plt.figure(figsize=(10, 5))

store_sales.plot(kind="bar")

plt.title("Total Sales by Store")
plt.xlabel("Store")
plt.ylabel("Sales")

plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# %%
# TOP ITEMS

item_sales = (
    df.groupby("item")["sales"]
      .sum()
      .sort_values(ascending=False)
)

top_items = item_sales.head(15)

plt.figure(figsize=(12, 6))

top_items.plot(kind="bar")

plt.title("Top 15 Items by Total Sales")
plt.xlabel("Item")
plt.ylabel("Sales")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# %%
# TIME-BASED FEATURE ENGINEERING

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["week"] = df["date"].dt.isocalendar().week.astype(int)
df["day"] = df["date"].dt.day
df["day_of_week"] = df["date"].dt.dayofweek
df["day_of_year"] = df["date"].dt.dayofyear
df["quarter"] = df["date"].dt.quarter

df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(int)

# Cyclical month representation
df["sin_month"] = np.sin(
    2 * np.pi * df["month"] / 12
)

df["cos_month"] = np.cos(
    2 * np.pi * df["month"] / 12
)

# Cyclical day-of-week representation
df["sin_dayofweek"] = np.sin(
    2 * np.pi * df["day_of_week"] / 7
)

df["cos_dayofweek"] = np.cos(
    2 * np.pi * df["day_of_week"] / 7
)


# %%
# LAG FEATURES

group_columns = ["store", "item"]

for lag in [1, 7, 14, 28]:

    df[f"lag_{lag}"] = (
        df.groupby(group_columns)["sales"]
          .shift(lag)
    )


# %%
# ROLLING DEMAND FEATURES

for window in [7, 14, 28]:

    df[f"rolling_mean_{window}"] = (
        df.groupby(group_columns)["sales"]
          .transform(
              lambda x:
              x.shift(1)
               .rolling(window)
               .mean()
          )
    )

    df[f"rolling_std_{window}"] = (
        df.groupby(group_columns)["sales"]
          .transform(
              lambda x:
              x.shift(1)
               .rolling(window)
               .std()
          )
    )


# %%
# REMOVE RECORDS CREATED WITH MISSING LAG VALUES

df = df.dropna().reset_index(drop=True)

print("Dataset after feature engineering:")
print(df.shape)


# %%
# ENCODE STORE AND ITEM

df["store"] = (
    df["store"]
    .astype("category")
    .cat.codes
)

df["item"] = (
    df["item"]
    .astype("category")
    .cat.codes
)


# %%
# SELECT MODEL FEATURES

feature_columns = [
    "store",
    "item",
    "year",
    "month",
    "week",
    "day",
    "day_of_week",
    "day_of_year",
    "quarter",
    "is_weekend",
    "sin_month",
    "cos_month",
    "sin_dayofweek",
    "cos_dayofweek",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_std_7",
    "rolling_std_14",
    "rolling_std_28"
]

X = df[feature_columns]
y = df["sales"]

print("Number of model features:", len(feature_columns))


# %%
# TIME-BASED TRAIN / VALIDATION SPLIT

validation_days = 30

cutoff_date = (
    df["date"].max()
    - pd.Timedelta(days=validation_days)
)

train_df = (
    df[df["date"] <= cutoff_date]
    .copy()
)

valid_df = (
    df[df["date"] > cutoff_date]
    .copy()
)

X_train = train_df[feature_columns]
y_train = train_df["sales"]

X_valid = valid_df[feature_columns]
y_valid = valid_df["sales"]

print("Training period:")
print(train_df["date"].min(), "to", train_df["date"].max())

print("\nValidation period:")
print(valid_df["date"].min(), "to", valid_df["date"].max())

print("\nTraining records:", len(train_df))
print("Validation records:", len(valid_df))


# %%
# TRAIN MACHINE LEARNING MODEL

model = HistGradientBoostingRegressor(
    learning_rate=0.08,
    max_iter=250,
    max_leaf_nodes=31,
    min_samples_leaf=30,
    l2_regularization=1.0,
    random_state=RANDOM_STATE
)

model.fit(
    X_train,
    y_train
)

print("Model training completed successfully.")


# %%
# GENERATE PREDICTIONS

valid_predictions = model.predict(
    X_valid
)

# Demand cannot be negative
valid_predictions = np.maximum(
    valid_predictions,
    0
)

forecast_results = valid_df[
    ["date", "store", "item", "sales"]
].copy()

forecast_results["predicted_sales"] = (
    valid_predictions
)

forecast_results["error"] = (
    forecast_results["sales"]
    - forecast_results["predicted_sales"]
)

forecast_results["absolute_error"] = (
    forecast_results["error"].abs()
)

print(forecast_results.head(10))


# %%
# MODEL EVALUATION

mae = mean_absolute_error(
    y_valid,
    valid_predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_valid,
        valid_predictions
    )
)

mape = (
    mean_absolute_percentage_error(
        y_valid,
        valid_predictions
    )
    * 100
)

print("=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"MAPE : {mape:.2f}%")


# %%
# ACTUAL VS PREDICTED SALES

comparison = (
    forecast_results
    .groupby("date")[
        ["sales", "predicted_sales"]
    ]
    .sum()
)

plt.figure(figsize=(14, 5))

plt.plot(
    comparison.index,
    comparison["sales"],
    label="Actual"
)

plt.plot(
    comparison.index,
    comparison["predicted_sales"],
    label="Predicted"
)

plt.title("Actual vs Predicted Daily Demand")
plt.xlabel("Date")
plt.ylabel("Sales")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# %%
# FORECAST ERROR ANALYSIS

error_by_item = (
    forecast_results
    .groupby("item")["absolute_error"]
    .mean()
    .sort_values(ascending=False)
)

print("Items with highest average forecast error:")

print(
    error_by_item
    .head(15)
    .to_frame("Mean Absolute Error")
)


# %%
# INVENTORY PLANNING

inventory = (
    forecast_results
    .groupby(["store", "item"])
    .agg(
        avg_daily_demand=(
            "predicted_sales",
            "mean"
        ),

        demand_std=(
            "predicted_sales",
            "std"
        ),

        actual_demand=(
            "sales",
            "sum"
        )
    )
    .reset_index()
)

inventory["demand_std"] = (
    inventory["demand_std"]
    .fillna(0)
)


# %%
# INVENTORY ASSUMPTIONS

lead_time_days = 7

# Approximate Z-score for ~95% service level
service_level_z = 1.65

inventory["safety_stock"] = (
    service_level_z
    * inventory["demand_std"]
    * np.sqrt(lead_time_days)
)

inventory["reorder_point"] = (
    inventory["avg_daily_demand"]
    * lead_time_days
    + inventory["safety_stock"]
)

# Approximate 30-day replenishment requirement
inventory["recommended_order_qty"] = (
    inventory["avg_daily_demand"]
    * 30
)


# %%
# INVENTORY PRIORITY CLASSIFICATION

inventory["inventory_priority"] = pd.qcut(
    inventory["recommended_order_qty"],
    q=3,
    labels=["Low", "Medium", "High"],
    duplicates="drop"
)

priority_summary = (
    inventory["inventory_priority"]
    .value_counts()
    .rename_axis("Priority")
    .reset_index(name="SKU_Count")
)

print(priority_summary)


# %%
# TOP REPLENISHMENT REQUIREMENTS

top_replenishment = (
    inventory
    .sort_values(
        "recommended_order_qty",
        ascending=False
    )
    .head(20)
)

print(
    top_replenishment[
        [
            "store",
            "item",
            "avg_daily_demand",
            "safety_stock",
            "reorder_point",
            "recommended_order_qty",
            "inventory_priority"
        ]
    ]
)


# %%
# VISUALIZE REPLENISHMENT REQUIREMENTS

plot_data = (
    top_replenishment
    .head(10)
    .copy()
)

plot_data["SKU"] = (
    "Store "
    + plot_data["store"].astype(str)
    + " - Item "
    + plot_data["item"].astype(str)
)

plt.figure(figsize=(12, 6))

plt.bar(
    plot_data["SKU"],
    plot_data["recommended_order_qty"]
)

plt.title(
    "Top 10 Recommended Monthly Order Quantities"
)

plt.xlabel("Store-Item")
plt.ylabel("Recommended Units")

plt.xticks(rotation=45)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()
plt.show()


# %%
# TOP DEMAND SKUS

high_demand_skus = (
    inventory
    .sort_values(
        "avg_daily_demand",
        ascending=False
    )
    .head(10)
)

print("TOP 10 HIGH-DEMAND STORE-ITEM COMBINATIONS")

print(
    high_demand_skus[
        [
            "store",
            "item",
            "avg_daily_demand",
            "safety_stock",
            "reorder_point"
        ]
    ]
)


# %%
# BUSINESS INSIGHTS

top_items = (
    df.groupby("item")["sales"]
      .sum()
      .sort_values(ascending=False)
      .head(10)
)

print("=" * 65)
print("KEY BUSINESS INSIGHTS")
print("=" * 65)

print("\nTop 10 items by historical demand:")
print(top_items)

print("\nTop store-item combinations by forecasted demand:")

print(
    high_demand_skus[
        [
            "store",
            "item",
            "avg_daily_demand",
            "reorder_point"
        ]
    ]
)


# %%
# EXPORT FORECAST RESULTS

forecast_results.to_csv(
    OUTPUT_DIR / "forecast_results.csv",
    index=False
)

print(
    "Forecast results saved to:",
    OUTPUT_DIR / "forecast_results.csv"
)


# %%
# EXPORT INVENTORY PLAN

inventory.to_csv(
    OUTPUT_DIR / "inventory_plan.csv",
    index=False
)

print(
    "Inventory plan saved to:",
    OUTPUT_DIR / "inventory_plan.csv"
)


# %%
# EXPORT EXCEL REPORT

excel_path = (
    OUTPUT_DIR
    / "retail_demand_analysis.xlsx"
)

with pd.ExcelWriter(
    excel_path,
    engine="openpyxl"
) as writer:

    forecast_results.to_excel(
        writer,
        sheet_name="Forecast Results",
        index=False
    )

    inventory.to_excel(
        writer,
        sheet_name="Inventory Plan",
        index=False
    )

    priority_summary.to_excel(
        writer,
        sheet_name="Priority Summary",
        index=False
    )

print(
    "Excel report saved to:",
    excel_path
)


# %%
# FINAL PROJECT SUMMARY

print("\n")
print("=" * 70)
print("RETAIL DEMAND FORECASTING & INVENTORY PLANNING")
print("=" * 70)

print(f"Historical records      : {len(df):,}")
print(f"Stores                  : {df['store'].nunique():,}")
print(f"Items                   : {df['item'].nunique():,}")
print(f"Validation records      : {len(valid_df):,}")

print("\nMODEL PERFORMANCE")
print(f"MAE                     : {mae:.2f}")
print(f"RMSE                    : {rmse:.2f}")
print(f"MAPE                    : {mape:.2f}%")

print("\nINVENTORY PLANNING")
print(f"SKUs analyzed           : {len(inventory):,}")
print(f"Lead time assumption    : {lead_time_days} days")
print(f"Service-level Z score   : {service_level_z}")

print("\nOUTPUT FILES")
print("- forecast_results.csv")
print("- inventory_plan.csv")
print("- retail_demand_analysis.xlsx")

print("\nProject completed successfully.")
