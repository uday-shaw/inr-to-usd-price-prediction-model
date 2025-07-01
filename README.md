#  inr-to-dollar-prediction-model
group project created for the sake of introduction to machine learning course CSL2010
# INR to USD Price Prediction Model

This repository contains machine learning models for predicting the exchange rate between the Indian Rupee (INR) and the US Dollar (USD), using both ARIMA (time series forecasting) and LSTM (deep learning) approaches. The project includes data preprocessing, model training, evaluation, and visualization for both methods.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Team Members](#team-members)
- [Files and Structure](#files-and-structure)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Results and Evaluation](#results-and-evaluation)
- [Future Work](#future-work)
- [License](#license)

---

## Project Overview

**Objective:**  
Predict the closing price of the INR/USD exchange rate using historical data. The project compares two approaches: traditional time series forecasting (ARIMA) and deep learning (LSTM).

**Key Features:**
- **ARIMA Model:** Implements a classic time series model for exchange rate prediction.
- **LSTM Model:** Uses a Long Short-Term Memory neural network to capture complex patterns in sequential data.
- **Exploratory Data Analysis:** Visualizes price distributions, trends, and correlations.
- **Model Evaluation:** Provides performance metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and visual comparisons of predictions.

---

## Team Members

- **Rishi Raj Bhagat (B23MT1034)**
- **Uday Shaw (B23CH1045)**
- **Sanjiban Sarkar (B23CH1041)**
- **Patel Nisarg Rakeshkumar (B23MT1029)**

---

## Files and Structure

### `algo_1_arima.py`
**Description:**  
Implements an ARIMA model for time series forecasting of the INR/USD exchange rate.

**Key Features:**
- **Data Loading:** Loads historical exchange rate data from `usdinr_d_2.csv`.
- **Train/Test Split:** Splits data into training (83%) and testing (17%) sets.
- **ARIMA Model:** Fits an ARIMA model (order=(5,1,0)) and predicts future prices.
- **Performance Evaluation:** Computes Mean Squared Error (MSE) between actual and predicted values.
- **Visualization:** Plots actual vs. predicted values and saves the graph as `arima_tp.png`[1].

---

### `project_file-1.ipynb`
**Description:**  
Jupyter notebook implementing an LSTM (Long Short-Term Memory) neural network for INR/USD price prediction.

**Key Steps:**
- **Data Loading and Preprocessing:** Loads `HistoricalData.csv`, processes dates, and selects the closing price as the target variable.
- **Data Scaling:** Normalizes data using MinMax scaling.
- **Exploratory Data Analysis:** Visualizes price distributions, trends, boxplots, and correlation heatmaps.
- **Data Preparation for LSTM:** Creates sequences of past prices as input features for the LSTM.
- **LSTM Model Building:** Constructs a sequential neural network with LSTM and dense layers.
- **Model Training:** Trains the model using Adam optimizer and mean squared error loss, with early stopping to prevent overfitting.
- **Model Evaluation:** Evaluates model performance on the test set and inverse-transforms predictions for comparison.
- **Visualization:** Plots actual vs. predicted prices and future price predictions[4].

---

### `Group-51-Project-Report.pdf`
**Description:**  
Project report detailing the motivation, methodology, data analysis, model architecture, training process, and results.

**Key Sections:**
- **Introduction:** Project objective and dataset description.
- **Code Overview and Structure:** Data loading, preprocessing, scaling, and exploratory analysis.
- **LSTM Model:** Architecture, training, early stopping, prediction, and evaluation.
- **Exploratory Data Analysis:** Visualizations and insights.
- **Conclusion and Future Work:** Summary of findings and suggestions for improvement[2].

---

### `ML-Project-PPT.pptx`
**Description:**  
Presentation slides summarizing the project, including:
- **Team Members**
- **Objective and Dataset**
- **Approach and Model Overview**
- **Forex Market Context**
- **LSTM Architecture and Training**
- **Results and Visualization**
- **Conclusion and Future Directions**[3].

---

## Setup Instructions

1. **Python Environment:**  
   Ensure you have Python 3.x installed. It is recommended to use a virtual environment.
2. **Dependencies:**  
   Install required libraries:
