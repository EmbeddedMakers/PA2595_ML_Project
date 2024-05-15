import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # to visualize the data features




import os, logging, sys


def data_load():
    # Input data files are available in the read-only "../input/" directory
    for dirname, _, filenames in os.walk('kaggle\input'):
        for filename in filenames:
            logging.debug(os.path.join(dirname, filename))
            dataset = pd.read_csv(os.path.join(dirname, filename))
            dataset.drop(columns=['Unnamed: 0'],axis=1,inplace=True)
            logging.debug(dataset.describe())
            logging.debug(dataset.head(5))
            return dataset



def data_plotting(dataset):
    # Plot price against each feature
    for each in dataset.columns.values:
        if each != 'final_price' and dataset[each].dtype!='O':
            plt.plot(dataset[each],dataset['final_price'] ,'o',)
            plt.title(f"{each} with Final Price")
            plt.xlabel(f"{each}")
            plt.ylabel("final price")
            plt.tight_layout()
           
            plt.show()

def main():
    print ("Hello")
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    dataset = data_load()
    data_plotting(dataset)
    
        
            

if __name__ == "__main__":
    main()