# Databricks notebook source
!pip install numpy pandas matplotlib seaborn scikit-learn xgboost hyperopt shap scipy


# COMMAND ----------

import os
OUTPUT_DIR = './' 
os.makedirs(OUTPUT_DIR, exist_ok=True)

# COMMAND ----------

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
from xgboost import XGBRegressor
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import shap

import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Configure plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("✓ All packages imported successfully!")
print(f"✓ NumPy version: {np.__version__}")
print(f"✓ Pandas version: {pd.__version__}")
print(f"✓ XGBoost version: {xgb.__version__}")
print(f"✓ SHAP version: {shap.__version__}")
print(f"\n✓ Output directory: {os.path.abspath(OUTPUT_DIR)}")

# COMMAND ----------

start_date = datetime(2020, 1, 1)
end_date = datetime(2023, 12, 31)
date_range = pd.date_range(start=start_date, end=end_date, freq='D')
n_days = len(date_range)

# Create base dataframe
df = pd.DataFrame({
    'region': ['NY'] * n_days,
    'city': ['New York'] * n_days,
    'date': date_range,
    'year': date_range.year,
    'month': date_range.month,
    'day': date_range.day,
    'day_of_week': date_range.dayofweek
})

print(f"✓ Generated {n_days} days of data")
print(f"✓ Date range: {start_date.date()} to {end_date.date()}")
print(f"✓ DataFrame shape: {df.shape}")
df.head()

# COMMAND ----------

base_traffic = 600

# Weekly seasonality (weekends have higher traffic)
weekly_pattern = np.sin(2 * np.pi * df['day_of_week'] / 7) * 150

# Monthly trend (gradual increase over time)
monthly_trend = np.linspace(0, 200, n_days)

# Random daily noise
random_noise = np.random.normal(0, 50, n_days)

# COVID-19 impact (sharp drop in March 2020)
covid_impact = np.where(df['date'] >= '2020-03-15', -400, 0)

# Combine all components
df['num_visits'] = (base_traffic + weekly_pattern + monthly_trend + 
                    random_noise + covid_impact).clip(lower=50)
df['num_visits'] = df['num_visits'].astype(int)

print("✓ Foot traffic data generated successfully!")
print(f"  - Min visits: {df['num_visits'].min()}")
print(f"  - Max visits: {df['num_visits'].max()}")
print(f"  - Mean visits: {df['num_visits'].mean():.0f}")
print(f"  - Std visits: {df['num_visits'].std():.0f}")

# Banner Impressions
df['banner_imp'] = np.around(
    np.random.randint(200000, 800000, n_days) * np.log(df['num_visits']) / 10
).astype(int)

# Social Media Likes
df['social_media_like'] = np.around(
    np.random.lognormal(3, 0.25, n_days) * df['num_visits'] / 1000
).astype(int)

# Landing Page Visits
raw_landing = np.around(
    np.random.lognormal(6, 0.03, n_days) * df['num_visits'] / 555
)
df['landing_page_visit'] = pd.Series(raw_landing).rolling(
    window=7, min_periods=1
).mean().fillna(400).astype(int)

# Google Trends
weeks = pd.date_range(start=start_date, end=end_date, freq='W')
google_trend_weekly = np.random.randint(50, 100, len(weeks))
google_trend_df = pd.DataFrame({
    'date': weeks,
    'google_trend': google_trend_weekly
})
df = df.merge(google_trend_df, on='date', how='left')
df['google_trend'] = df['google_trend'].fillna(method='ffill').fillna(82).astype(int)

print("✓ All campaign metrics generated")
print(f"  - Banner impressions: {df['banner_imp'].min():,} to {df['banner_imp'].max():,}")
print(f"  - Social media likes: {df['social_media_like'].min()} to {df['social_media_like'].max()}")
print(f"  - Landing page visits: {df['landing_page_visit'].min()} to {df['landing_page_visit'].max()}")
print(f"  - Google trends: {df['google_trend'].min()} to {df['google_trend'].max()}")


# COMMAND ----------

df.to_csv(os.path.join(OUTPUT_DIR, 'foot_traffic_data.csv'), index=False)
print(f"✓ Dataset saved to: {os.path.join(OUTPUT_DIR, 'foot_traffic_data.csv')}")
print(f"\nSample of generated data:")
display(df.head(10))

fig, axes = plt.subplots(5, 1, figsize=(18, 14))
fig.suptitle('Campaign Metrics Over Time', fontsize=16, fontweight='bold')

axes[0].plot(df['date'], df['num_visits'], color='#2E86AB', linewidth=1.5)
axes[0].set_ylabel('Store Visits', fontsize=11, fontweight='bold')
axes[0].set_title('Daily Store Visits', fontsize=12)
axes[0].grid(True, alpha=0.3)

axes[1].bar(df['date'], df['banner_imp'], color='#F77F00', width=1, alpha=0.7)
axes[1].set_ylabel('Banner Impressions', fontsize=11, fontweight='bold')
axes[1].set_title('Banner Impressions', fontsize=12)
axes[1].grid(True, alpha=0.3, axis='y')

axes[2].bar(df['date'], df['social_media_like'], color='#06A77D', width=1, alpha=0.7)
axes[2].set_ylabel('Social Media Likes', fontsize=11, fontweight='bold')
axes[2].set_title('Social Media Likes', fontsize=12)
axes[2].grid(True, alpha=0.3, axis='y')

axes[3].plot(df['date'], df['google_trend'], color='#D62828', linewidth=2)
axes[3].fill_between(df['date'], df['google_trend'], alpha=0.3, color='#D62828')
axes[3].set_ylabel('Google Trend', fontsize=11, fontweight='bold')
axes[3].set_title('Google Trend Score', fontsize=12)
axes[3].grid(True, alpha=0.3)

axes[4].plot(df['date'], df['landing_page_visit'], color='#7209B7', linewidth=1.5)
axes[4].set_ylabel('Landing Page Visits', fontsize=11, fontweight='bold')
axes[4].set_title('Landing Page Visits', fontsize=12)
axes[4].set_xlabel('Date', fontsize=12, fontweight='bold')
axes[4].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'timeseries_plot.png'), dpi=300, bbox_inches='tight')
plt.show()
print(f"✓ Saved: timeseries_plot.png")


correlation_features = ['num_visits', 'banner_imp', 'social_media_like', 
                        'landing_page_visit', 'google_trend']
correlation_matrix = df[correlation_features].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            fmt='.3f', square=True, linewidths=1)
plt.title('Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
plt.show()
print(f"✓ Saved: correlation_matrix.png")

print("\nCorrelations with Store Visits:")
correlations = df[correlation_features].corr()['num_visits'].sort_values(ascending=False)
for feature, corr in correlations.items():
    if feature != 'num_visits':
        print(f"  {feature:25s}: {corr:6.3f}")


feature_cols = ['banner_imp', 'social_media_like', 'landing_page_visit', 'google_trend']
target_col = 'num_visits'

X = df[feature_cols].copy()
y = df[target_col].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=55
)

print("="*80)
print("DATA PREPARED FOR MACHINE LEARNING")
print("="*80)
print(f"✓ Features: {feature_cols}")
print(f"✓ Target: {target_col}")
print(f"✓ Training set: {X_train.shape[0]} samples")
print(f"✓ Test set: {X_test.shape[0]} samples")

def objective(params):
    model = XGBRegressor(
        learning_rate=params['learning_rate'],
        gamma=params['gamma'],
        max_depth=int(params['max_depth']),
        n_estimators=int(params['n_estimators']),
        min_child_weight=int(params['min_child_weight']),
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )
    
    cv_scores = cross_val_score(
        model, X_train, y_train, 
        cv=5, 
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    score = -cv_scores.mean()
    return {'loss': score, 'status': STATUS_OK}

search_space = {
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
    'gamma': hp.uniform('gamma', 0, 5),
    'max_depth': hp.quniform('max_depth', 3, 10, 1),
    'n_estimators': hp.quniform('n_estimators', 50, 300, 10),
    'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1)
}

print("✓ Objective function and search space defined")


# COMMAND ----------


print("="*80)
print("RUNNING HYPERPARAMETER OPTIMIZATION")
print("="*80)
print("This may take 3-5 minutes...\n")

trials = Trials()
best_params = fmin(
    fn=objective,
    space=search_space,
    algo=tpe.suggest,
    max_evals=30,
    trials=trials,
    rstate=np.random.default_rng(42),
    verbose=1
)

best_params['max_depth'] = int(best_params['max_depth'])
best_params['n_estimators'] = int(best_params['n_estimators'])
best_params['min_child_weight'] = int(best_params['min_child_weight'])

print("\n✓ Optimization complete!")
print("\nBest Parameters:")
for param, value in best_params.items():
    print(f"  {param:20s}: {value}")


final_model = XGBRegressor(
    learning_rate=best_params['learning_rate'],
    gamma=best_params['gamma'],
    max_depth=best_params['max_depth'],
    n_estimators=best_params['n_estimators'],
    min_child_weight=best_params['min_child_weight'],
    objective='reg:squarederror',
    random_state=42,
    n_jobs=-1
)

final_model.fit(X_train, y_train)
print("✓ Model trained successfully!")


y_train_pred = final_model.predict(X_train)
y_test_pred = final_model.predict(X_test)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

print("="*80)
print("MODEL PERFORMANCE")
print("="*80)
print(f"Train R²:   {train_r2:.4f}")
print(f"Test R²:    {test_r2:.4f}")
print(f"Train RMSE: {train_rmse:.2f}")
print(f"Test RMSE:  {test_rmse:.2f}")
print(f"Train MAE:  {train_mae:.2f}")
print(f"Test MAE:   {test_mae:.2f}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': final_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
display(feature_importance)

df['pred_num_visits'] = final_model.predict(X)

plt.figure(figsize=(20, 7))
plt.plot(df['date'], df['num_visits'], label='Actual', color='#2E86AB', linewidth=2)
plt.plot(df['date'], df['pred_num_visits'], label='Predicted', 
         color='#F77F00', linestyle='--', linewidth=2)
plt.xlabel('Date', fontsize=13, fontweight='bold')
plt.ylabel('Store Visits', fontsize=13, fontweight='bold')
plt.title('Predicted vs Actual Foot Traffic', fontsize=15, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'predictions_vs_actual.png'), dpi=300, bbox_inches='tight')
plt.show()
print(f"✓ Saved: predictions_vs_actual.png")

# COMMAND ----------

print("="*80)
print("SHAP ANALYSIS")
print("="*80)
print("Calculating SHAP values (this may take a minute)...")

explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X)

mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Mean |SHAP|': mean_abs_shap
}).sort_values('Mean |SHAP|', ascending=False)

print("\n✓ SHAP analysis complete!")
print("\nChannel Effectiveness (Mean |SHAP|):")
display(shap_importance)


# Bar plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.title('Feature Contribution (SHAP)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'shap_summary_bar.png'), dpi=300, bbox_inches='tight')
plt.show()
print(f"✓ Saved: shap_summary_bar.png")

# Beeswarm plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X, show=False)
plt.title('SHAP Value Distribution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'shap_summary_beeswarm.png'), dpi=300, bbox_inches='tight')
plt.show()
print(f"✓ Saved: shap_summary_beeswarm.png")

# ============================================================================
# CELL 17: SHAP Attribution DataFrame
# ============================================================================
shap_df = pd.DataFrame(shap_values, columns=feature_cols)
shap_df['base_value'] = explainer.expected_value
shap_df['date'] = df['date'].values
shap_df['actual_visits'] = df['num_visits'].values
shap_df['predicted_visits'] = df['pred_num_visits'].values

shap_df.to_csv(os.path.join(OUTPUT_DIR, 'shap_attribution.csv'), index=False)
print(f"✓ Saved: shap_attribution.csv")
print("\nSample SHAP attributions:")
display(shap_df.head(10))

# ============================================================================
# CELL 18: Business Insights and Recommendations
# ============================================================================
print("\n" + "="*80)
print("KEY BUSINESS INSIGHTS")
print("="*80)

top_channel = shap_importance.iloc[0]['Feature'].replace('_', ' ').title()
second_channel = shap_importance.iloc[1]['Feature'].replace('_', ' ').title()

print(f"\n📊 TOP FINDINGS:")
print(f"\n1. MOST EFFECTIVE CHANNEL: {top_channel}")
print(f"   Mean Impact: {shap_importance.iloc[0]['Mean |SHAP|']:.2f}")
print(f"   Recommendation: Increase investment in this channel")

print(f"\n2. SECOND MOST EFFECTIVE: {second_channel}")
print(f"   Mean Impact: {shap_importance.iloc[1]['Mean |SHAP|']:.2f}")
print(f"   Recommendation: Maintain or boost campaigns")

print(f"\n3. MODEL PERFORMANCE:")
print(f"   Test R²: {test_r2:.3f}")
print(f"   Test RMSE: {test_rmse:.2f} visits")
print(f"   Insight: Model provides reliable predictions")

print("\n💡 RECOMMENDATIONS:")
print(f"   ✓ Shift budget to {top_channel}")
print(f"   ✓ Test variations in top 2 channels")
print(f"   ✓ Use predictions for staffing/inventory")
print(f"   ✓ Monitor SHAP values for campaign effectiveness")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"\n✓ All files saved to: {os.path.abspath(OUTPUT_DIR)}")
print("\nGenerated files:")
print("  • foot_traffic_data.csv")
print("  • shap_attribution.csv")
print("  • timeseries_plot.png")
print("  • correlation_matrix.png")
print("  • predictions_vs_actual.png")
print("  • shap_summary_bar.png")
print("  • shap_summary_beeswarm.png")

print("\n🎉 Ready for business decisions! 🎉")


# COMMAND ----------

# DBTITLE 1,Databricks Visualization
import pandas as pd
DATA_DIR = './campaign_effectiveness_data/'
GOLD_PATH = f'{DATA_DIR}gold/'

# 2. Load the data safely
try:
    df_for_dashboard = gold_data
except NameError:
    print(f"Variable not found. Loading from {GOLD_PATH}...")
    df_for_dashboard = pd.read_parquet(f'{GOLD_PATH}foot_traffic_gold.parquet')

# 3. Convert to Spark for Databricks Dashboarding
gold_spark_df = spark.createDataFrame(df_for_dashboard)

# 4. Display for Dashboard
display(gold_spark_df)

# COMMAND ----------

table_name = "campaign_effectiveness_gold"

gold_spark_df.write \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(table_name)

print(f"✓ Table '{table_name}' created successfully. You can now build a dashboard on it.")