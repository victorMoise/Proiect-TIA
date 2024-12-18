import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
from memory_profiler import memory_usage

def main():
  df = pd.read_csv('lego_dataset.csv')

  df['date_released'] = pd.to_datetime(df['date_released'])
  df['year_released'] = df['date_released'].dt.year

  df = df.drop(columns=['date_released', 'genre'])

  X = df.drop(columns=['units_sold'])
  y = df['units_sold']

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  model_random_forest = RandomForestRegressor(random_state=42)
  model_linear_regression = LinearRegression()

  # random forest training metrics
  start_time = time.time()
  mem_usage_rf = memory_usage((model_random_forest.fit, (X_train, y_train)))
  train_time_rf = time.time() - start_time

  # random forest prediction time metrics 
  start_time = time.time()
  y_test_pred = model_random_forest.predict(X_test)
  predict_time_rf = time.time() - start_time

  # linear regression training metrics 
  start_time = time.time()
  mem_usage_lr = memory_usage((model_linear_regression.fit, (X_train, y_train)))
  train_time_lr = time.time() - start_time

  # linear regression prediction metrics
  start_time = time.time()
  y_test_pred_linear = model_linear_regression.predict(X_test)
  predict_time_lr = time.time() - start_time

  # random forest performance indicators
  mse_random_forest = mean_squared_error(y_test, y_test_pred)
  rmse_random_forest = mean_squared_error(y_test, y_test_pred, squared=False)
  mae_random_forest = mean_absolute_error(y_test, y_test_pred)
  r2_random_forest = 1 - r2_score(y_test, y_test_pred)
  
  # linear regression performance indicators
  mse_linear_regression = mean_squared_error(y_test, y_test_pred_linear)
  r2_linear_regression = 1 - r2_score(y_test, y_test_pred_linear)
  mae_linear_regression = mean_absolute_error(y_test, y_test_pred_linear)
  rmse_linear_regression = mean_squared_error(y_test, y_test_pred_linear, squared=False)

  # display random forest results
  print("Random Forest:")
  print("Mean Squared Error:", mse_random_forest)
  print("R-squared:", r2_random_forest)
  print("Mean Absolute Error:", mae_random_forest)
  print("Root Mean Squared Error:", rmse_random_forest)
  print("Training Time:", train_time_rf, "seconds")
  print("Prediction Time:", predict_time_rf, "seconds")
  print("Memory Usage:", max(mem_usage_rf) - min(mem_usage_rf), "MB")

  # display linear regression results
  print("\nLinear Regression:")
  print("Mean Squared Error:", mse_linear_regression)
  print("R-squared:", r2_linear_regression)
  print("Mean Absolute Error:", mae_linear_regression)
  print("Root Mean Squared Error:", rmse_linear_regression)
  print("Training Time:", train_time_lr, "seconds")
  print("Prediction Time:", predict_time_lr, "seconds")
  print("Memory Usage:", max(mem_usage_lr) - min(mem_usage_lr), "MB")

  # plot actual vs predicted values for random forest
  plt.figure(figsize=(10, 6))
  plt.plot(y_test.values, label='Actual Values', color='blue')
  plt.plot(y_test_pred, label='Predicted Values (Random Forest)', color='red', linestyle='--')
  plt.title('Actual vs Predicted Values (Random Forest)')
  plt.xlabel('Index')
  plt.ylabel('Pieces Sold')
  plt.legend()
  plt.show()

  # plot actual vs predicted values for linear regression
  plt.figure(figsize=(10, 6))
  plt.plot(y_test.values, label='Actual Values', color='blue')
  plt.plot(y_test_pred_linear, label='Predicted Values (Linear Regression)', color='green', linestyle='--')
  plt.title('Actual vs Predicted Values (Linear Regression)')
  plt.xlabel('Index')
  plt.ylabel('Pieces Sold')
  plt.legend()
  plt.show()


if __name__ == '__main__':
  main()