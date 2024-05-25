import pandas as pd
import numpy as np
from data_processing import train_models, data_load, data_prepare
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

dataset = data_load()

X, y = data_prepare(dataset)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

best_model, second_model = train_models(X_train, y_train)

y_pred = best_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

# Print evaluation metrics
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')
