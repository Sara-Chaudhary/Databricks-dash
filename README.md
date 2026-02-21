# 📈 Databricks Sales Forecasting & Advertising Attribution
### *An End-to-End Implementation with Custom Synthetic Data*

## 📌 Project Overview
Inspired by the **Databricks Solution Accelerator**, this project explores the relationship between marketing spend, search intent, and physical store visits. 
Most implementations use pre-existing datasets; however, for this project, I **engineered a custom synthetic dataset** to simulate real-world noise, regional seasonality, and marketing correlations across 10 major U.S. states.

## 🚀 Key Features
* **Data Synthesis:** Developed a custom engine to generate millions of rows of data, including `num_visit`, `ad_spend`, `Google Search_trends`, and `social_media_likes`.
* **Streamlined Architecture:** Consolidated the standard 5-notebook Databricks workflow into **two high-efficiency Python files**.
* **Regional Analytics:** Performed granular analysis on 10 states (NY, CA, TX, FL, etc.) to identify top-performing markets.
* **Correlation Mapping:** Established the statistical link between "Banner Impressions" and "Total Foot Traffic."

## 🛠 Tech Stack
* **Platform:** Databricks
* **Language:** Python (PySpark, Pandas, NumPy)
* **Storage:** Delta Lake
* **Visualization:** Databricks SQL Dashboards
* **Methodology:** Medallion Architecture (Bronze -> Silver -> Gold)

## 📂 Project Structure
1.  **`Data_Generation_Engine.py`**: The "source of truth." This script simulates three years of daily marketing and sales data (2020–2023), injecting specific trends and correlations.
2.  **`Analysis_Pipeline.py`**: The transformation layer. Cleans raw data, calculates attribution metrics, and prepares the "Gold" tables for the dashboard.

   
## 📊 Dashboard Highlights
The final **Campaign Effectiveness Dashboard** provides:
* **Total Foot Traffic:** 3.63M visits tracked.
* **Ad Value:** Managed $4.99M in simulated ad impression value.
* **Trend Identification:** Visualized how Google Search Trends serve as a leading indicator for store visits.
* **Regional Volume:** New York identified as the highest volume contributor.
<img width="1749" height="869" alt="Dashboard (1)" src="https://github.com/user-attachments/assets/a6d42d63-e24b-4b76-8f05-7e8ea275a0b1" />

---
*Developed as an independent project to demonstrate full-stack Data Engineering and Analytics capabilities on the Databricks Lakehouse Platform.*

