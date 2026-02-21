Sales Forecasting & Advertising Attribution
Self-Generated Data Project
📌 Project Overview
This project is a custom implementation inspired by the Databricks Sales Forecasting Solution Accelerator. I developed a synthetic dataset to simulate real-world marketing conditions and built a comprehensive attribution pipeline to connect advertising spend with store performance.

🚀 Technical Efficiency
Unlike standard implementations, this project was optimized into a streamlined two-file architecture:

Data_Engine.py: A custom Python script that generates synthetic sales and marketing data, incorporating seasonal trends, regional variations (10 states), and noise to simulate real-world complexity.

Analytics_Pipeline.py: A unified script that handles data cleaning, correlation analysis between search intent and foot traffic, and the final logic for the executive dashboard.

<img width="1749" height="869" alt="Dashboard (1)" src="https://github.com/user-attachments/assets/a6d42d63-e24b-4b76-8f05-7e8ea275a0b1" />
📊 Key Insights from Dashboard
The Campaign Effectiveness Dashboard I built reveals:
Total Impact: Managed and visualized 3.63M in total foot traffic and $4.99M in ad impression value.
Leading Indicators: Identified that Google Trends acts as a leading indicator for physical store visits.
Marketing Correlation: Established a strong positive correlation between Banner Impressions and actual store visit spikes.
Regional Performance: Visualized performance across 10 major US regions (NY, CA, TX, etc.), identifying NY as the volume leader.

🛠 Tech Stack
Platform: Databricks
Language: Python (PySpark)
Data Strategy: Synthetic Data Generation (Numpy/Pandas)
Analytics: Delta Lake & Databricks SQL
