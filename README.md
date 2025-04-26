# Ex.No: 6               HOLT WINTERS METHOD
### Date:26-04-2025 



### AIM:
To implement Holt-Winters model on Onion Price Data Set and make future predictions
### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```py
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd

data = pd.read_csv('/content/OnionTimeSeries - Sheet1.csv', index_col='Date', parse_dates=True)

data = data['Min'].resample('MS').mean()

# Scaling the Data using MinMaxScaler 
scaler = MinMaxScaler()
data_scaled = pd.Series(scaler.fit_transform(data.values.reshape(-1, 1)).flatten(), index=data.index)

# Split into training and testing sets (80% train, 20% test)
train_data = data_scaled[:int(len(data_scaled) * 0.8)]
test_data = data_scaled[int(len(data_scaled) * 0.8):]

fitted_model_add = ExponentialSmoothing(
    train_data, trend='add', seasonal='add', seasonal_periods=12
).fit()

# Forecast and evaluate
test_predictions_add = fitted_model_add.forecast(len(test_data))

# Evaluate performance
print("MAE :", mean_absolute_error(test_data, test_predictions_add))
print("RMSE :", mean_squared_error(test_data, test_predictions_add, squared=False))

# Plot predictions
plt.figure(figsize=(12, 8))
plt.plot(train_data, label='TRAIN', color='black')
plt.plot(test_data, label='TEST', color='green')
plt.plot(test_predictions_add, label='PREDICTION', color='red')
plt.title('Train, Test, and Additive Holt-Winters Predictions')
plt.legend(loc='best')
plt.show()

final_model = ExponentialSmoothing(data, trend='mul', seasonal='mul', seasonal_periods=12).fit()

# Forecast future values
forecast_predictions = final_model.forecast(steps=12)

data.plot(figsize=(12, 8), legend=True, label='Current Price of Onion')
forecast_predictions.plot(legend=True, label='Forecasted Price of Onion')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title(' Forecast of Onion Price')
plt.show()
```

### OUTPUT:
EVALUATION

![image](https://github.com/user-attachments/assets/511bee3a-4f2e-49b0-92fc-bb199eed586a)

TEST_PREDICTION
![Untitled](https://github.com/user-attachments/assets/8e47c722-6a83-4d3d-9a16-2080a9e18283)


FINAL_PREDICTION
![Untitled-1](https://github.com/user-attachments/assets/2485d7ab-e568-4822-9fc5-2454f60e6195)


### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
