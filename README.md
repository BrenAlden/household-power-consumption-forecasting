# Household Power Consumption Prediction using LSTM

## Project Description

This project focuses on the forecasting of household power consumption using deep learning, specifically a Long Short-Term Memory (LSTM) neural network. The goal is to accurately predict future electric power usage based on historical time-series data. This automated forecasting system can assist households and energy providers in understanding consumption patterns, enabling them to optimize energy usage and management.

The project involves loading the household power consumption dataset via Kagglehub, preprocessing the data, and utilizing the LSTM model for sequential point prediction. Visualizations are employed to understand model performance during training.

## Expected Features

*   **Data Loading and Preprocessing:** Reading the dataset using `kagglehub`, combining date and time features into a datetime index, resampling the active power data to daily averages, and applying necessary preprocessing steps such as `MinMaxScaler` normalization.
*   **Data Visualization:** Displaying line plots to evaluate training versus validation loss across epochs.
*   **Model Building:** Utilizing an LSTM architecture adapted for time-series forecasting, featuring an LSTM layer with 50 units and a `TimeDistributed` dense layer on top.
*   **Model Training and Evaluation:** Training the model using the Adam optimizer and Mean Squared Error (MSE) loss function, while employing an `EarlyStopping` callback to prevent overfitting. Performance is also evaluated by comparing actual versus predicted values.

## Further Steps
*   **Feature Engineering:** Incorporating additional variables such as global reactive power, voltage, or sub-metering data to increase the accuracy of the forecasting model.
*   **Hyperparameter Tuning:** Experimenting with different learning rates, batch sizes, sequence time steps, and optimization algorithms to achieve optimal performance.
*   **Deployment:** Integrating the trained model into a web or mobile application for real-world energy monitoring.

---

# README: Household Power Consumption Prediction

This repository contains the code for a deep learning project aimed at forecasting household power consumption using an LSTM model.

## Overview

Accurate and timely prediction of energy consumption is crucial for efficient power management and cost reduction. This project leverages the power of Recurrent Neural Networks (RNNs), specifically the LSTM architecture, to automate the forecasting of power usage from historical consumption logs.

## Dataset

The project utilizes the `household-power-consumption` dataset sourced directly via `kagglehub`. The dataset contains measurements such as global active power, reactive power, voltage, and global intensity. The data is retrieved and cached automatically as a CSV file:

```text
/kaggle/input/household-power-consumption/
└── household_power_consumption.csv
```

## Dependencies

The following Python libraries are required to run the code:

*   Pandas
*   NumPy
*   OS
*   Matplotlib
*   Scikit-learn
*   TensorFlow / Keras
*   Kagglehub

You can install these dependencies using `pip`:

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow kagglehub
```

## Project Structure

*   `Household_Power_Consumption.ipynb`: This Jupyter Notebook contains the complete pipeline for the project.
    *   It defines the steps to download the dataset using Kagglehub.
    *   It handles EDA, preprocessing (resampling to daily data and scaling), and sequence generation.
    *   It builds, trains, and plots the loss history of the LSTM model.

## Getting Started

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    ```
2.  **Prepare the data:** The dataset is automatically handled by the `kagglehub.dataset_download("imtkaggleteam/household-power-consumption")` command within the notebook, so manual downloading is not required.
3.  **Run the code:** Execute the `Household_Power_Consumption.ipynb` Jupyter Notebook sequentially to perform data loading, model training, and prediction generation.

## Acknowledgements

*   This project utilizes the LSTM architecture from TensorFlow Keras layers.
*   Dataset provided by the IMT Kaggle Team via Kaggle.
