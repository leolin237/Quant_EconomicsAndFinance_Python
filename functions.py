""" 
Integration a function from using norm and calculate
"""
from scipy.stats import norm
from scipy.integrate import quad
phi = norm.pdf  # the function to integrate
value, error = quad(phi, -147, 147)


"""
Linear regression of Y = 20 + 4*X + error with error following a normal distribution N(0,25)
"""
import numpy as np
X = np.array([4,8,12,16,20])
t = np.array([0,1,2,3,4])
Y = 20 + 4*X + np.random.normal(0,25,5)
beta = np.polyfit(t,Y,1) # linear regression of Y = a + b*t with a and b the values beta[0] and beta[1]

"""
A Machine Learning Model from scikit-learn library
"""
import pandas as pd
melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)
melbourne_data.columns

# Prediction target
y = melbourne_data.Price

# Selection of the features for making our prediction model
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

# Selection of the features
X = melbourne_data[melbourne_features]

# Splitting the data into training and validation data, for both features and target
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
# Define model
from sklearn.tree import DecisionTreeRegressor
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(train_X, train_y)
# Get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)

# Mean Absolute Error
from sklearn.metrics import mean_absolute_error
val_mae = mean_absolute_error(val_predictions, val_y) # This is the error we want to minimize, it describes the quality of our model

# Overfitting and underfitting
"""
Since we care about accuracy on new data, which we estimate from our validation data, we want to find the sweet spot between 
underfitting and overfitting

Conclusion
Here's the takeaway: Models can suffer from either:

Overfitting: capturing spurious patterns that won't recur in the future, leading to less accurate predictions, or
Underfitting: failing to capture relevant patterns, again leading to less accurate predictions.
We use validation data, which isn't used in model training, to measure a candidate model's accuracy. This lets us try many candidate 
models and keep the best one.
"""
