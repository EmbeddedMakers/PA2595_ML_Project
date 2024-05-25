import pandas as pd
from data_processing import train_models, data_load, data_prepare

dataset = data_load()
X, _ = data_prepare(dataset)
best_model, second_model = train_models(X, dataset['final_price'])

# Prompt the user to enter house data
example_features = {}
for feature in X.columns:
    value = input(f"Enter the value for {feature}: ")
    example_features[feature] = float(value)

features_df = pd.DataFrame([example_features])
predicted_price = best_model.predict(features_df)[0]

# Print predicted price
print(f'Predicted House Price: {predicted_price}')
