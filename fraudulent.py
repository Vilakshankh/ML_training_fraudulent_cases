import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys



#Import data set
with open(sys.argv[1], 'r') as f:
    data = pd.read_csv(f , sep='\t')
print(data)


#Split our dataset into its attributes and labels
attributes = data.iloc[:, :-2].values #attributes
labels = data.iloc[:, -2].values #labels

from sklearn.model_selection import train_test_split
#train_test_split: Split arrays or matrices into random train and test subsets.
#Randomly select U=25 instances to be the instances in UnInstance
attributes_train, attributes_test, labels_train, labels_test = train_test_split(attributes, labels, test_size = 0.25)

 
from sklearn.preprocessing import StandardScaler
#scale the features so that all of them can be uniformly evaluated
scaler = StandardScaler()
scaler.fit(attributes_train)

attributes_train = scaler.transform(attributes_train)
attributes_test = scaler.transform(attributes_test)

from sklearn.linear_model import LogisticRegression

# Create a logistic regression model with default hyperparameters
model = LogisticRegression()

# Fit the model to the training data
model.fit(attributes_train, labels_train)

# Use the model to make predictions on the test data
labels_pred = model.predict(labels_test)




from sklearn.metrics import precision_score

# Calculate the precision of the model
precision = precision_score(labels_test, labels_pred)

print("Precision:", precision)