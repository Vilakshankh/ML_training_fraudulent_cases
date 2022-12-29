import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys



#Import data set
with open(sys.argv[1], 'r') as f:
    data = pd.read_csv(f , sep='\t')
print(data)


#Splitting

#Split our dataset into its attributes and labels
attributes = data.iloc[:, :-2].values #attributes
labels = data.iloc[:, -2].values #labels

from sklearn.model_selection import train_test_split
#train_test_split: Split arrays or matrices into random train and test subsets.
#Randomly select U=25 instances to be the instances in UnInstance
attributes_train, attributes_test, labels_train, labels_test = train_test_split(attributes, labels, test_size = 0.25)







# Encoding

from sklearn.preprocessing import LabelEncoder

# Create a LabelEncoder object
encoder = LabelEncoder()

# Fit the encoder to the categorical data for the nameOrig column
encoder.fit(data['nameOrig'])

# Transform the nameOrig column in the training and test sets
attributes_train[:, 3] = encoder.fit_transform(attributes_train[:, 3])
attributes_test[:, 3] = encoder.fit_transform(attributes_test[:, 3])

# Fit the encoder to the categorical data for the nameDest column
encoder.fit(data['nameDest'])

# Transform the nameDest column in the training and test sets
attributes_train[:, 5] = encoder.fit_transform(attributes_train[:, 5])
attributes_test[:, 5] = encoder.fit_transform(attributes_test[:, 5])




# Scaling

from sklearn.preprocessing import StandardScaler
# Select the columns to be scaled (exclude the nameOrig and nameDest columns)
attributes_train_scaled = attributes_train[:, [0, 1, 2, 4, 6, 7]]
attributes_test_scaled = attributes_test[:, [0, 1, 2, 4, 6, 7]]

# Scale the features
scaler = StandardScaler()
scaler.fit(attributes_train_scaled)

attributes_train_scaled = scaler.transform(attributes_train_scaled)
attributes_test_scaled = scaler.transform(attributes_test_scaled)





#Training

from sklearn.linear_model import LogisticRegression

# Create a logistic regression model with default hyperparameters
model = LogisticRegression()

# Fit the model to the training data
model.fit(attributes_train, labels_train)

# Use the model to make predictions on the test data
labels_pred = model.predict(labels_test)



#Precision

from sklearn.metrics import precision_score

# Calculate the precision of the model
precision = precision_score(labels_test, labels_pred)

print("Precision:", precision)