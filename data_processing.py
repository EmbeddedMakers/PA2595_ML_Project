import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # to visualize the data features




import os, logging, sys

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error, mean_absolute_error

def data_load():
    # Input data files are available in the read-only "../input/" directory
    for dirname, _, filenames in os.walk('kaggle\input'):
        for filename in filenames:
            logging.debug(os.path.join(dirname, filename))
            dataset = pd.read_csv(os.path.join(dirname, filename))
            # The column named 'Unnamed: 0' is dropped from the dataset.
            # The inplace=True parameter means that the operation is performed directly on the dataset without needing to reassign it.
            dataset.drop(columns=['Unnamed: 0'],axis=1,inplace=True)
            logging.debug(dataset.describe())
            logging.debug(dataset.head(5))
            return dataset

def data_prepare(dataset):
    """The purpose of this function is to prepare the data for a machine learning model:
        1. Create Features (X): A DataFrame X is created that contains only the numeric features from the original dataset, 
        with the target variable final_price removed.
        2. Create Target (Y): A Series Y is created that contains the final_price values.
    """
    # Step 1: Create a features dataset
    features = ['asked_price', 'land_area', 'area', 'price_per_area', 'rooms', 'supplemental_area']
    X = dataset[features]
    # Step 2: Extract the target variable
    Y = dataset.final_price
    
    for each in X.columns:
        # Step 3: Print the count of non-null and null values for each column.This is needed for identifying missing values in the dataset and data cleaning.
        print(X[each].isnull().value_counts())
        # Step 4: Fill missing values with the mean of the column
        X[each] = X[each].fillna(X[each].mean())
    # Step 5: Print the count of non-null and null values for the target variable 'final_price'
    print(Y.isnull().value_counts())
    # Step 6: Fill missing values in the target variable with the mean of the column
    Y = Y.fillna(Y.mean())
    # Step 7: Return the features and target variable
    return X, Y

def data_plotting(dataset):
    """The purpose of this function is to visualize the relationship between the numeric features in the dataset 
    and the target variable 'final_price'.
    """
    # Iterate over each column in the dataset
    for each in dataset.columns.values:
        # Check if the column is not 'final_price' and if it is not of object type (i.e., it's numeric)
        if each != 'final_price' and dataset[each].dtype != 'O':
            # Create a scatter plot for the current column against 'final_price'
            plt.plot(dataset[each], dataset['final_price'], 'o')
            # Set the title of the plot to indicate which column is being plotted
            plt.title(f"{each} with Final Price")
            # Label the x-axis with the current column name
            plt.xlabel(f"{each}")
            # Label the y-axis as 'final price'
            plt.ylabel("final price")
            # Adjust layout to ensure the plot elements fit well
            plt.tight_layout()
            # Display the plot
            plt.show()

def train_models(X_train, y_train):
    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }

    # Train and evaluate models using cross-validation
    results = {}
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        rmse_scores = np.sqrt(-cv_scores)
        results[name] = {'RMSE': rmse_scores.mean(), 'STD': rmse_scores.std()}

    results_df = pd.DataFrame(results).T
    results_df.sort_values(by='RMSE', inplace=True)
    print(results_df)
    
    # Get the name of the best model
    best_model_name = results_df.index[0]
    print(f'Best model name: {best_model_name}')
    
    # Select the best model from the models dictionary
    best_model = models[best_model_name]
    best_model.fit(X_train, y_train)
    
    # Get the name of the best model
    second_model_name = results_df.index[1]
    print(f'Second model name: {second_model_name}')
    
    # Select the best model from the models dictionary
    second_model = models[second_model_name]
    second_model.fit(X_train, y_train)
    
    return best_model, second_model


def main():
    print ("Hello")
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    dataset = data_load()
    # Describe function is used to generate descriptive statistics that summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding NaN values.
    dataset.describe()
    #data_plotting(dataset)
    x, y = data_prepare(dataset)
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    best_model, second_model = train_models(X_train, y_train)
    




if __name__ == "__main__":
    main()