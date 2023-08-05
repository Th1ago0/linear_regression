# Predicting the financial return on investments in public bonds

# Packages
from os import getcwd
from math import sqrt
from csv import reader, writer

# Get current working directory
path = getcwd()

# function to average
def average(values):
    return sum(values) / float(len(values))

# Function to calculate covariance
def covariance(x, x_mean, y, y_mean):
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i] - x_mean) * (y[i] - y_mean)
    return covar

# Function to calculate variance
def variance(list, mean):
    return sum([(x - mean) ** 2 for x in list])

# Function to calculate coefficient
def coefficient(covar, var, x_mean, y_mean):
    b1 = covar / var
    b0 = y_mean - (b1 * x_mean)
    return b1, b0

# Function to load data
def load_data(dataset):
    init = 0
    x = list()
    y = list()
    with open(dataset) as file:
        content = reader(file)
        for row in content:
            if init == 0:
                init = 1
            else:
                x.append(row[0])
                y.append(row[1])
    return x, y

# Function to split data
def split_dataset(x, y):
    x_train = list()
    x_test = list()
    y_train = list()
    y_test = list()

    training_size = int(.8 * len(x))

    x_train, x_test = x[0:training_size], x[training_size::]
    y_train, y_test = y[0:training_size], y[training_size::]

    return x_train, x_test, y_train, y_test

# Function to predict
def predict(b1, b0, x_test):
    predicted_y = list()
    for x in x_test:
        predicted_y.append(b0 + b1 * x)
    return predicted_y

# Function to calculate the rmse
def rmse(predicted_y, y_test):
    sum_error = 0.0
    for i in range(len(predicted_y)):
        sum_error = (predicted_y[i] - y_test[i]) ** 2
    return sqrt(sum_error / float(len(y_test)))

# Block main for execution
def main():

    try:
        # Load dataset
        dataset = path + '/datasets/dataset.csv'
        x, y = load_data(dataset)

        # Prepare data
        x = [float(i) for i in x]
        y = [float(i) for i in y]

        # Calculating the average values of x and y, covariance and variance
        x_mean = average(x)
        y_mean = average(y)
        covar = covariance(x, x_mean, y, y_mean)
        var = variance(x, x_mean)

        # Split data
        x_train, x_test, y_train, y_test = split_dataset(x, y)

        # Calculate the coefficients (training)
        b1, b0 = coefficient(covar, var, x_mean, y_mean)
        print(f'Coefficients\nB0: {b0} B1: {b1}')

        # Predicts with the model
        predicted_y = predict(b1, b0, x_test)

        # Model errors
        root_mean = rmse(predicted_y, y_test)

        print(f'\nLinear regression model without frameworks')
        print(f'Average model error: {root_mean}')

        # New data
        new_x = float(input('Enter the investment amount: '))

        # Predicts
        new_y = b0 + b1 * new_x
        print(f'Predict: {new_y}\n')

    except Exception as er:
        print(er)

main()