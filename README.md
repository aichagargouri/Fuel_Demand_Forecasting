## Files

##### Dataset used: is the dataset used to predict the fuel demand at Afriquia stations.
##### demand-forecasting (1).ipynb: This notebook contains the code for predicting the fuel demand using the models and includes the process of model training, tuning, and evaluation.
##### comparative-data.ipynb: This notebook is dedicated to comparing the prediction results from the demand-forecasting notebook with the actual data. It also handles additional data not taken into account in the demand-forecasting notebook for more comprehensive insights.
##### Fuel demand prediction for Afriquia station.pdf: this file is the final form of the report on the Fuel demand prediction project.

# Fuel Demand Forecasting Using Machine Learning and Deep Learning

## Overview
This project aims to predict fuel demand at Afriquia stations using various predictive models, including machine learning, deep learning, and time series models. The dataset used was aggregated by station and day, comprising 8,911 rows and 11 features after preprocessing. The project explores model performance, evaluates predictions, and compares their accuracy based on standard evaluation metrics.

## Objectives
Forecast daily fuel demand at Afriquia stations.
Compare the performance of machine learning, deep learning, and time series models.
Implement feature engineering to capture cyclical patterns and optimize model accuracy.
Data
The original dataset consisted of over 6.7 million rows and was aggregated by station and day to create a smaller, manageable dataset for forecasting. Feature engineering was applied to create cyclic features, such as sine and cosine transformations, to capture trends in the data over time (e.g., daily, weekly, monthly).

## Exploratory Data Analysis (EDA)
#### Aggregation: The dataset was aggregated by day and station, with insights into daily fuel consumption patterns.
#### Visualizations: Time series plots, histograms, and correlation matrices were used to explore data distributions, relationships, and anomalies.
#### Clustering: DBSCAN clustering revealed that the data was uniform, forming only a single cluster.
#### Cyclical Features: Features like Month_sin, Day_sin, and WeekOfYear_cos were created to capture cyclical trends in fuel demand, which were important for time series forecasting.

## Model Development
Various models were tested for fuel demand prediction:

### Machine Learning Models:
#### Gradient Boosting Machine (GBM): This model performed the best, providing highly accurate predictions and effectively balancing error metrics.
#### Random Forest (RF): RF also performed well, closely following GBM in terms of accuracy.
#### XGBoost: Although powerful, XGBoost did not outperform GBM and RF in this case.
#### Decision Trees (DT): DT was the weakest among the machine learning models, showing higher errors and lower prediction accuracy.

### Deep Learning Models:
#### RNN, LSTM, GRU: These deep learning models underperformed on the dataset and exhibited clear signs of underfitting, leading to their exclusion from final comparisons.

### Time Series Models:
#### ARIMA, SARIMA: Time series models struggled to make accurate predictions, with significant errors due to underfitting.

## Model Tuning and Optimization
Hyperparameter optimization was carried out using Optuna across all machine learning models to improve accuracy and efficiency. Each model’s performance was fine-tuned by testing various configurations, especially for GBM and Random Forest.

## Evaluation and Results
#### The models were evaluated using:

##### Mean Squared Error (MSE)
##### Root Mean Squared Error (RMSE)
##### Mean Absolute Error (MAE)
##### R-squared (R²)

### Best Models:
#### GBM emerged as the best model, achieving the highest accuracy and lowest error rates.
#### Random Forest (RF) followed closely behind GBM in performance.
#### Decision Trees (DT) performed the worst among the machine learning models, with significantly higher error rates.
#### Due to the poor performance of deep learning and time series models, only the machine learning models with better predictions were included in the final comparison.

## Conclusion
This project demonstrated the effectiveness of machine learning models, particularly GBM and Random Forest, in predicting fuel demand at Afriquia stations. These models can help the company optimize its operations and improve decision-making processes.

## Future Work
Potential improvements include:

##### Incorporating more external data to enhance the predictive power of the models.
##### Applying advanced ensemble techniques to further optimize performance.
