# Databricks notebook source
# Install required packages
!pip install pandas numpy pyarrow -q

# COMMAND ----------

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("✓ Libraries imported successfully")

# COMMAND ----------

# Configuration
DATA_DIR = './campaign_effectiveness_data/'
BRONZE_PATH = f'{DATA_DIR}bronze/'
SILVER_PATH = f'{DATA_DIR}silver/'
GOLD_PATH = f'{DATA_DIR}gold/'

# Create directories
import os
for path in [DATA_DIR, BRONZE_PATH, SILVER_PATH, GOLD_PATH]:
    os.makedirs(path, exist_ok=True)
    
print(f"✓ Directory structure created at: {DATA_DIR}")

# COMMAND ----------

def generate_high_volume_safegraph_data(avg_stores_per_city=5):
    np.random.seed(42)
    data = []
    # 1. Expanded Lists to increase combinations
    brands = [
        'Subway', 'Dunkin\'', 'Starbucks', 'McDonald\'s', 'Target', 
        'Walmart', 'CVS', 'Walgreens', 'Home Depot', 'Whole Foods'
    ]
    
    cities = [
        ('New York', 'NY'), ('Newark', 'NJ'), ('Philadelphia', 'PA'),
        ('Chicago', 'IL'), ('Houston', 'TX'), ('Phoenix', 'AZ'),
        ('Los Angeles', 'CA'), ('Seattle', 'WA'), ('Miami', 'FL'), ('Boston', 'MA')
    ]
    
    # 2. Timeframe: 14 Months (Jan 2019 - Feb 2020)
    start_date = '2020-01-01'
    end_date = '2023-12-31'
    
    # Generate list of month start dates
    periods = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    months = []
    for dt in periods:
        days_in_m = dt.days_in_month
        end_dt = dt + pd.Timedelta(days=days_in_m - 1)
        
        months.append((
            dt.strftime('%Y%m'),          # Format: 202001
            dt.strftime('%Y-%m-%d'),      # Start: 2020-01-01
            end_dt.strftime('%Y-%m-%d'),  # End: 2020-01-31
            days_in_m                     # Duration: 31
        ))
    
    base_place_id = 100000
    place_id_counter = 0
        
    # Iterate through Cities and Brands
    for city, region in cities:
        for brand in brands:
            
            # Randomize store count (e.g., if avg is 5, generate between 3 and 7 stores)
            num_stores = np.random.randint(max(1, avg_stores_per_city - 2), avg_stores_per_city + 3)
            
            for i in range(num_stores):
                # Create a unique ID for this specific physical store
                current_place_id = f"sg-{base_place_id + place_id_counter}"
                place_id_counter += 1
                
                # Generate a consistent address for this store
                street_num = np.random.randint(100, 9999)
                street_name = np.random.choice(['Main', 'Broadway', 'Market', 'Wall', 'Broad', 'Maple'])
                address = f"{street_num} {street_name} St"
                
                # Generate 14 months of data for this SINGLE store
                for year_month, start_date, end_date, num_days in months:
                    
                    # Simulate seasonality: Higher traffic in Dec/July, lower in Feb
                    base_traffic = np.random.randint(200, 800)
                    if '12' in year_month or '07' in year_month:
                        base_traffic += 200
                        
                    # Generate daily visits with some noise
                    visits = (np.random.poisson(base_traffic / num_days, num_days) + 
                              np.random.randint(10, 50, num_days)).tolist()
                    
                    data.append({
                        'safegraph_place_id': current_place_id,
                        'location_name': brand,
                        'street_address': address,
                        'city': city,
                        'region': region,
                        'date_range_start': int(pd.Timestamp(start_date).timestamp()),
                        'date_range_end': int(pd.Timestamp(end_date).timestamp()),
                        'year_month': year_month,
                        'visits_by_day': json.dumps(visits)
                    })
    
    return pd.DataFrame(data)

# --- Run the generation with a multiplier ---
# Setting avg_stores_per_city=5 means about 500 unique stores total
df_large = generate_high_volume_safegraph_data(avg_stores_per_city=5)

print(f"--- Generation Complete ---")
print(f"Total Rows Generated: {len(df_large):,}")
print(f"Total Unique Stores:  {df_large['safegraph_place_id'].nunique()}")
print(f"Total Brands:         {df_large['location_name'].nunique()}")
print(f"Total Cities:         {df_large['city'].nunique()}")

# Show a sample
df_large.head()

# COMMAND ----------

# Save to Bronze layer
bronze_file = f'{BRONZE_PATH}foot_traffic_bronze.parquet'
df_large.to_parquet(bronze_file, index=False)
print(f"✓ Bronze data saved to: {bronze_file}")
print(f"  Shape: {df_large.shape}")

# COMMAND ----------

def explode_visits_by_day(df):
    """
    Explode visits_by_day array into separate rows
    Each row will represent one day of foot traffic
    """
    exploded_data = []
    
    for _, row in df.iterrows():
        visits_array = json.loads(row['visits_by_day'])
        
        # Extract year and month from timestamp
        start_date = pd.Timestamp(row['date_range_start'], unit='s')
        year = start_date.year
        month = start_date.month
        
        # Determine MSA (Metropolitan Statistical Area)
        msa = 'NYC MSA' if row['region'] in ['NY', 'PA', 'NJ'] else 'US'
        
        # Create a row for each day in the month
        for day_idx, num_visits in enumerate(visits_array):
            exploded_data.append({
                'safegraph_place_id': row['safegraph_place_id'],
                'location_name': row['location_name'],
                'city': row['city'],
                'region': row['region'],
                'msa': msa,
                'year': year,
                'month': month,
                'day': day_idx + 1,  # 1-indexed
                'num_visits': num_visits,
                'date_range_start': row['date_range_start'],
                'date_range_end': row['date_range_end']
            })
    
    return pd.DataFrame(exploded_data)

# Transform the data
visits_by_day = explode_visits_by_day(df_large)
print(f"✓ Exploded to {len(visits_by_day)} daily records")
visits_by_day.head(10)

# COMMAND ----------

# Save to Silver layer
silver_file = f'{SILVER_PATH}foot_traffic_silver.parquet'
visits_by_day.to_parquet(silver_file, index=False)
print(f"✓ Silver data saved to: {silver_file}")
print(f"  Shape: {visits_by_day.shape}")
print(f"  Date range: {visits_by_day['year'].min()}-{visits_by_day['month'].min()} to {visits_by_day['year'].max()}-{visits_by_day['month'].max()}")

# COMMAND ----------

# 1. Create a proper date column in the Silver data first
visits_by_day['date'] = pd.to_datetime(visits_by_day[['year', 'month', 'day']])

# 2. Aggregate by Date, Brand, AND City
aggregated_data = visits_by_day.groupby(
    ['date', 'location_name', 'city', 'region']
)['num_visits'].sum().reset_index()

# 3. Sort for cleaner plotting
aggregated_data = aggregated_data.sort_values(['location_name', 'city', 'date']).reset_index(drop=True)

print(f"✓ Data Aggregated: {len(aggregated_data):,} rows")
print(f"  Brands included: {aggregated_data['location_name'].unique()}")
print(f"  Cities included: {aggregated_data['city'].unique()}")
aggregated_data.head()

# COMMAND ----------

np.random.seed(42)

# 1. Banner Impressions (Correlated with visits + noise)
aggregated_data['banner_imp'] = np.around(
    (aggregated_data['num_visits'] * np.random.uniform(200, 400)) + 
    np.random.normal(0, 500, len(aggregated_data))
).astype(int).clip(lower=0)

# 2. Social Media Likes (Lognormal distribution)
aggregated_data['social_media_like'] = np.around(
    np.random.lognormal(3, 0.5, len(aggregated_data)) * (aggregated_data['num_visits'] / 300)
).astype(int)

# 3. Landing Page Visits (7-Day Moving Average PER BRAND)
aggregated_data['landing_page_visit'] = (
    aggregated_data.groupby(['location_name', 'city'])['num_visits']
    .transform(lambda x: x.rolling(window=7, min_periods=1).mean())
)

print("✓ Generated marketing features.")

# --- COMPATIBILITY FIX ---
nyc_subway_agg = aggregated_data
nyc_subway_agg.head()

print("✓ Generated campaign media features:")
print(f"  - banner_imp: {nyc_subway_agg['banner_imp'].min():,} to {nyc_subway_agg['banner_imp'].max():,}")
print(f"  - social_media_like: {nyc_subway_agg['social_media_like'].min()} to {nyc_subway_agg['social_media_like'].max()}")
print(f"  - landing_page_visit: {nyc_subway_agg['landing_page_visit'].min():.0f} to {nyc_subway_agg['landing_page_visit'].max():.0f}")

# COMMAND ----------

# Generate Google Trends data (weekly data that will be upsampled to daily)
def generate_google_trends():
    start_date = pd.Timestamp('2019-01-01') 
    end_date = pd.Timestamp('2023-12-31') # Updated to match your 4-year scope
    
    # Create weekly range
    weeks = pd.date_range(start=start_date, end=end_date, freq='W-SUN')
        
    t = np.arange(len(weeks))
    seasonality = 15 * np.sin(2 * np.pi * t / 52)
    trend = t * 0.1 
    noise = np.random.normal(0, 5, len(weeks))
    trends = 60 + seasonality + trend + noise
    
    # Clip to keep between 0 and 100 (Google Trends scale)
    trends = np.clip(trends, 0, 100).astype(int)
    
    return pd.DataFrame({
        'date': weeks,
        'google_trend': trends
    })

google_trends = generate_google_trends()
print(f"✓ Generated {len(google_trends)} weeks of REALISTIC Google Trends data")
google_trends.head()

# COMMAND ----------

 # Merge Google Trends and forward-fill to upsample from weekly to daily
gold_data = nyc_subway_agg.merge(google_trends, on='date', how='left')
gold_data['google_trend'] = gold_data['google_trend'].ffill().fillna(82).astype(int)

# Format date as string
gold_data['date'] = gold_data['date'].dt.strftime('%Y-%m-%d')

print(f"✓ Gold dataset ready with {len(gold_data)} records")
print(f"\nFeature summary:")
print(gold_data[['num_visits', 'banner_imp', 'social_media_like', 'landing_page_visit', 'google_trend']].describe())

# COMMAND ----------

# Display final Gold dataset
print("\n📊 Final Gold Dataset (First 20 rows):")
gold_data.head(20)

# COMMAND ----------

# Save to Gold layer
gold_file = f'{GOLD_PATH}foot_traffic_gold.parquet'
gold_data.to_parquet(gold_file, index=False)
print(f"✓ Gold data saved to: {gold_file}")
print(f"  Shape: {gold_data.shape}")
print(f"  Columns: {list(gold_data.columns)}")

# COMMAND ----------

# Install plotting libraries if needed
!pip install matplotlib seaborn -q

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 8)

# Convert date back to datetime for plotting
plot_data = gold_data.copy()
plot_data['date'] = pd.to_datetime(plot_data['date'])

# COMMAND ----------

# Plot 1: Foot Traffic Over Time
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(plot_data['date'], plot_data['num_visits'], linewidth=1.5, color='#1f77b4')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Number of Visits', fontsize=12)
ax.set_title('Subway Foot Traffic Nationwide(Jan 2020 - Dec 2023)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{GOLD_PATH}foot_traffic_timeseries.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: foot_traffic_timeseries.png")

# COMMAND ----------

# Plot 2: Campaign Media Features
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Banner Impressions
axes[0, 0].plot(plot_data['date'], plot_data['banner_imp'], color='#ff7f0e', linewidth=1)
axes[0, 0].set_title('Banner Impressions', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Impressions')
axes[0, 0].grid(True, alpha=0.3)

# Social Media Likes
axes[0, 1].plot(plot_data['date'], plot_data['social_media_like'], color='#2ca02c', linewidth=1)
axes[0, 1].set_title('Social Media Likes', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Likes')
axes[0, 1].grid(True, alpha=0.3)

# Landing Page Visits
axes[1, 0].plot(plot_data['date'], plot_data['landing_page_visit'], color='#d62728', linewidth=1)
axes[1, 0].set_title('Landing Page Visits (7-day MA)', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Visits')
axes[1, 0].set_xlabel('Date')
axes[1, 0].grid(True, alpha=0.3)

# Google Trends
axes[1, 1].plot(plot_data['date'], plot_data['google_trend'], color='#9467bd', linewidth=1)
axes[1, 1].set_title('Google Trends Index', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Search Interest')
axes[1, 1].set_xlabel('Date')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{GOLD_PATH}campaign_media_features.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: campaign_media_features.png")

# COMMAND ----------

# Plot 3: Correlation Heatmap
fig, ax = plt.subplots(figsize=(10, 8))
correlation_matrix = plot_data[['num_visits', 'banner_imp', 'social_media_like', 'landing_page_visit', 'google_trend']].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f'{GOLD_PATH}correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: correlation_heatmap.png")

# COMMAND ----------

# Generate summary report
print("="*80)
print("CAMPAIGN EFFECTIVENESS ETL PIPELINE - SUMMARY REPORT")
print("="*80)
print(f"\n📁 Data Layers:")
print(f"  Bronze (Raw):        {len(df_large):,} rows")
print(f"  Silver (Daily):      {len(visits_by_day):,} rows")
print(f"  Gold (Enriched):     {len(gold_data):,} rows")

print(f"\n📅 Date Coverage:")
print(f"  Start: {gold_data['date'].min()}")
print(f"  End:   {gold_data['date'].max()}")
print(f"  Days:  {len(gold_data)} days")

print(f"\n📊 Feature Statistics:")
print(f"  Foot Traffic:")
print(f"    Mean:   {gold_data['num_visits'].mean():.0f} visits/day")
print(f"    Median: {gold_data['num_visits'].median():.0f} visits/day")
print(f"    Range:  {gold_data['num_visits'].min()}-{gold_data['num_visits'].max()} visits/day")

print(f"\n  Campaign Media:")
print(f"    Banner Impressions:   {gold_data['banner_imp'].mean():,.0f} (avg)")
print(f"    Social Media Likes:   {gold_data['social_media_like'].mean():.0f} (avg)")
print(f"    Landing Page Visits:  {gold_data['landing_page_visit'].mean():.0f} (avg)")
print(f"    Google Trends:        {gold_data['google_trend'].mean():.0f} (avg)")

print(f"\n💾 Output Files:")
print(f"  {bronze_file}")
print(f"  {silver_file}")
print(f"  {gold_file}")
print(f"  {GOLD_PATH}foot_traffic_timeseries.png")
print(f"  {GOLD_PATH}campaign_media_features.png")
print(f"  {GOLD_PATH}correlation_heatmap.png")

print("\n" + "="*80)
print("✓ ETL Pipeline completed successfully!")
print("="*80)

# COMMAND ----------

# Export to CSV for easy loading in other tools
csv_file = f'{GOLD_PATH}foot_traffic_gold.csv'
gold_data.to_csv(csv_file, index=False)
print(f"✓ Exported to CSV: {csv_file}")

# Also create a data dictionary
data_dict = {
    'region': 'US state code (NY = New York)',
    'year': 'Year of observation',
    'month': 'Month of observation (1-12)',
    'day': 'Day of month',
    'num_visits': 'Total foot traffic (number of visits)',
    'date': 'Date in YYYY-MM-DD format',
    'banner_imp': 'Banner ad impressions (campaign media)',
    'social_media_like': 'Social media engagement count',
    'landing_page_visit': 'Website landing page visits (7-day moving average)',
    'google_trend': 'Google Trends search interest index (0-100)'
}

dict_df = pd.DataFrame(list(data_dict.items()), columns=['Column', 'Description'])
dict_file = f'{GOLD_PATH}data_dictionary.csv'
dict_df.to_csv(dict_file, index=False)
print(f"✓ Data dictionary saved: {dict_file}")
print("\nData Dictionary:")
print(dict_df.to_string(index=False))