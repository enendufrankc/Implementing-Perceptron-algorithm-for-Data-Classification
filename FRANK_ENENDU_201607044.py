### Import Relevant Libraries
import numpy as np # Numerical computing library
import pandas as pd # Data manipulation library
import matplotlib.pyplot as plt # Plotting library
"""
    QUESTION TWO
"""
#### Train Algorithm
def perceptronTrain(D, maxIter):
    """
    Trains a perceptron model using the given training data.

    Parameters:
        D (list): A list of tuples representing the training data. Each tuple contains an input feature vector (X) and its corresponding label (y).
        maxIter (int): The maximum number of iterations to train the model.

    Returns:
        tuple: A tuple containing the bias term (b) and weight vector (W) learned by the perceptron model.
    """
    # Initialize weight vector and bias term
    W = [0.0] * len(D[0][0])
    b = 0.0

    # Loop over training set for a maximum number of iterations
    for iter in range(maxIter):
        # Loop over all training examples
        for X, y in D:
            # Compute activation
            activation_function = sum([W[i] * X[i] for i in range(len(X))]) + b
            # Update weights and bias if prediction is incorrect
            if y * activation_function <= 0:
                W = [int(W[i] + y * X[i]) for i in range(len(X))]
                b += y

    # Return learned parameters
    return b, W


### Test Algorithm
def perceptronTest(b, W, X):
    """
    This function takes in a bias term `b`, a weight vector `W`, and a feature vector `X`.
    It calculates the weighted sum of the inputs plus the bias term, and returns 1 if the sum
    is greater than 0, otherwise it returns -1.
    
    Args:
    - b (float): The bias term for the perceptron model.
    - W (list): The weight vector for the perceptron model.
    - X (list): The feature vector for a single example.
    
    Returns:
    - int: Returns 1 if the weighted sum of inputs plus the bias term is greater than 0, 
           otherwise returns -1.
    """
    # Calculate the weighted sum of the inputs plus the bias term
    a = sum([W[i] * X[i] for i in range(len(X))]) + b
    # If the sum is greater than 0, return 1, otherwise return -1
    return 1 if a > 0 else -1

"""
    QUESTION THREE
"""
### Read Dataset Function
def readData(f_path):
    """
    Read data from a csv file located at the given file_path, convert class labels to integers and return as a list of tuples.

    Args:
        file_path (str): The path of the csv file.

    Returns:
        list: A list of tuples, where each tuple contains a list of feature values and its corresponding class label.
    """
    # Read data from the csv file located at the given file_path without considering the first row as header
    data = pd.read_csv(f_path, header=None)
    # Convert the class labels in the 5th column of the dataframe into integers, where 1 represents "class-1", 2 represents "class-2" and 3 represents "class-3"
    data[4] = data[4].apply(lambda x: 1 if x == "class-1" else 2 if x == "class-2" else 3)
    # Convert the dataframe into a list of tuples, where each tuple contains a list of feature values and its corresponding class label
    data = data.apply(lambda x: (x[:4].tolist(), int(x[4])), axis=1).tolist()
    # Return the list of tuples
    return data

# Read the training data from the "train.data" file using the readData() function
df_train = readData("train.data")
# Read the test data from the "test.data" file using the readData() function
df_test = readData("test.data")

### Data Manipulation Function
def dataPrep(data, class_label, multiclass=False):
    """
    This function takes in a dataset, a class label, and a boolean flag indicating whether the problem is multiclass or binary. It returns a modified dataset in the form of a list of tuples, where each tuple contains a list of feature values and a corresponding class label.
    For a multiclass problem, the function labels samples belonging to the target class with 1 and samples belonging to other classes with -1. For a binary problem, the function excludes samples belonging to the target class from the modified dataset.

    Parameters:
        data: A list of tuples, where each tuple contains a list of feature values and a corresponding class label
        class_label: The target class label
        multiclass: A boolean flag indicating whether the problem is multiclass or binary. Default value is False.
    
    Returns:
        new_data: A modified dataset in the form of a list of tuples, where each tuple contains a list of feature values and a corresponding class label.
    """
    # Create an empty list to store the modified dataset
    new_data = []
    
    # Iterate over each sample in the original dataset
    for X, y in data:
        # If the problem is multiclass
        if multiclass:
            # If the sample belongs to the target class, label it with 1
            if y == class_label:
                new_data.append((X, 1))
            # Otherwise, label it with -1
            else:
                new_data.append((X, -1))
        # If the problem is binary
        else:
            # If the sample belongs to the target class, exclude it from the modified dataset
            if y != class_label:
                new_data.append((X, y))
    
    # If no samples belong to the target class, return an empty list
    if not new_data:
        return []
    # Otherwise, continue modifying the dataset
    else:
            min_label = min([y for _, y in new_data])
            return [(X, 1 if y == min_label else -1) for X, y in new_data]
    
### Accuracy Calculation Function
def accuracy(train, test, maxIter):
    """
    Calculate the accuracy of a perceptron model on the training and test data.

    Args:
    train (list): The training data as a list of tuples, where each tuple contains the features and label.
    test (list): The test data as a list of tuples, where each tuple contains the features and label.
    maxIter (int): The maximum number of iterations to run the perceptron algorithm.

    Returns:
    str: A formatted string containing the training accuracy and test accuracy.
    """
    X_train = [d[0] for d in train]
    y_train = [d[-1] for d in train]
    X_test = [d[0] for d in test]
    y_test = [d[-1] for d in test]

    # Initialize an empty list to store the accuracy for each regularization coefficient
    accuracy = []

    # Train a perceptron model on the training data
    b, w = perceptronTrain(train, maxIter)

    # Make predictions on the training and test data using the trained model
    ytrainpred = [perceptronTest(b, w, x) for x in X_train]
    ytestpred = [perceptronTest(b, w, x) for x in X_test]

    # Calculate accuracy on the training data
    correct_train = sum(1 for a, b in zip(y_train, ytrainpred) if a == b)
    train_accuracy = round(correct_train/len(y_train) * 100, 2)

    # Calculate accuracy on the test data
    correct_test = sum(1 for a, b in zip(y_test, ytestpred) if a == b)
    test_accuracy = round(correct_test/len(y_test) * 100, 2)

    # Return the train and test accuracies as a formatted string
    return f"Traind Acuracy is: {train_accuracy}%, Test Accuracy is: {test_accuracy}%"

print("QUESTION THREE\n")
### Discriminate between class 1 and class 2,
# Preparing binary training and testing data for classes 1 and 2
binary_data_class1_2_train = dataPrep(df_train, 3, multiclass=False)
binary_data_class1_2_test = dataPrep(df_test, 3, multiclass=False)

print(f" Weight and Bias for Class1-2 Discrimination: {perceptronTrain(binary_data_class1_2_train, 20)}")
binary_class1_2_accuracy = accuracy(binary_data_class1_2_train, binary_data_class1_2_test, 20)
print(f"{binary_class1_2_accuracy}\n")

### Discriminate between class 2 and class 3,
# Preparing binary training and testing data for classes 2 and 3
binary_data_class2_3_train = dataPrep(df_train, 1, multiclass=False)
binary_data_class2_3_test = dataPrep(df_test, 1, multiclass=False)

print(f" Weight and Bias for Class2-3 Discrimination: {perceptronTrain(binary_data_class2_3_train, 20)}")
binary_class2_3_accuracy = accuracy(binary_data_class2_3_train, binary_data_class2_3_test, 20)
print(f"{binary_class2_3_accuracy}\n")

### Discriminate between class 1 and class 3,
# Preparing binary training and testing data for classes 1 and 3
binary_data_class1_3_train = dataPrep(df_train, 2, multiclass=False)
binary_data_class1_3_test = dataPrep(df_test, 2, multiclass=False)

print(f" Weight and Bias for Class1-3 Discrimination: {perceptronTrain(binary_data_class1_3_train, 20)}")
binary_class1_3_accuracy = accuracy(binary_data_class1_3_train, binary_data_class1_3_test, 20)
print(f"{binary_class1_3_accuracy}\n")

"""
    QUESTION FOUR
"""

print("QUESTION FOUR\n")
### Multiclass for 1 vs rest
# Preparing binary training and testing data for Multiclass for 1 vs rest
multi_class1vsrest_train = dataPrep(df_train, 1, multiclass=True)
multi_class1vsrest_test = dataPrep(df_test, 1, multiclass=True)

print(f" Weight and Bias for Multiclass for 1 vs rest: {perceptronTrain(multi_class1vsrest_train, 20)}")
Multi_class1vsrest_accuracy = accuracy(multi_class1vsrest_train, multi_class1vsrest_test, 20)
print(f"{Multi_class1vsrest_accuracy}\n")

### Multiclass for 2 vs rest
# Preparing binary training and testing data for Multiclass for 2 vs rest
multi_class2vsrest_train = dataPrep(df_train, 2, multiclass=True)
multi_class2vsrest_test = dataPrep(df_test, 2, multiclass=True)

print(f" Weight and Bias for Multiclass for 2 vs rest: {perceptronTrain(multi_class2vsrest_train, 20)}")
Multi_class2vsrest_accuracy = accuracy(multi_class2vsrest_train, multi_class2vsrest_test, 20)
print(f"{Multi_class2vsrest_accuracy}\n")

### Multiclass for 3 vs rest
# Preparing binary training and testing data for Multiclass for 3 vs rest
multi_class3vsrest_train = dataPrep(df_train, 3, multiclass=True)
multi_class3vsrest_test = dataPrep(df_test, 3, multiclass=True)

print(f" Weight and Bias for Multiclass for 3 vs rest: {perceptronTrain(multi_class3vsrest_train, 20)}")
multi_class3vsrest_accuracy = accuracy(multi_class3vsrest_train, multi_class3vsrest_test, 20)
print(f"{multi_class3vsrest_accuracy}\n")

"""
QUESTION FIVE
"""
#### Train Algorithm
def RegperceptronTrain(data, maxIter, regcoeffs):
    """Trains a regularized perceptron model on the input data.

    Args:
        data (list): A list of tuples containing feature vectors and labels.
        maxIter (int): The maximum number of iterations to run the perceptron training algorithm.
        regcoeffs (float): The regularization coefficient for the weight vector.

    Returns:
        tuple: A tuple containing the learned bias term and weight vector.
    """

    # Initialize weight vector and bias term
    W = [0.0] * len(data[0][0])
    b = 0.0

    # Loop over training set for a maximum number of iterations
    for iter in range(maxIter):
        # Loop over all training examples
        for X, y in data:
            # Compute activation
            activation_function = sum([W[i] * X[i] for i in range(len(X))]) + b
            # Update weights and bias if prediction is incorrect
            if y * activation_function <= 0:
                W = [(1 - 2*regcoeffs) * W[i] + y * X[i] for i in range(len(X))]
                b += y

    # Return learned parameters
    return b, W

# A list of regularisation coefficience
regCoeff = [0.01, 0.1, 1.0, 10.0, 100.0]

### Accuracy Calculation Function
def RegAccuracy(train, test, maxIter, regCoeff):
    """
    Calculates the accuracy of a perceptron model with varying regularization coefficients on the training and test datasets.
    
    Parameters:
    train (list): A list of tuples representing the training data. Each tuple contains the features as a list and the label as an integer.
    test (list): A list of tuples representing the testing data. Each tuple contains the features as a list and the label as an integer.
    maxIter (int): The maximum number of iterations to run the perceptron algorithm for.
    regCoeff (list): A list of regularization coefficients to test the perceptron model with.
    
    Returns:
    pandas.DataFrame: A DataFrame containing the regularization coefficients and corresponding train and test accuracies.
    """
    # function code here
    # Extract the features and labels from the training and testing datasets
    X_train = [d[0] for d in train]
    y_train = [d[-1] for d in train]
    X_test = [d[0] for d in test]
    y_test = [d[-1] for d in test]
    
    # Initialize an empty list to store the accuracy for each regularization coefficient
    accuracy = []
    
    # Loop through each regularization coefficient in regCoeff
    for coeff in regCoeff:
        # Train a perceptron model on the training dataset using the regularization coefficient and maximum number of iterations provided
        b, w = RegperceptronTrain(train, maxIter, coeff)
        
        # Make predictions on the training and testing datasets using the trained model
        ytrainpred = [perceptronTest(b, w, x) for x in X_train]
        ytestpred = [perceptronTest(b, w, x) for x in X_test]

        # Calculate the accuracy of the model's predictions for the training and testing datasets
        train_accuracy = sum([1 for i in range(len(y_train)) if y_train[i] == ytrainpred[i]]) / len(y_train) * 100
        test_accuracy = sum([1 for i in range(len(y_test)) if y_test[i] == ytestpred[i]]) / len(y_test) * 100
        
        # Store the regularization coefficient and corresponding accuracies in the accuracy list
        accuracy.append([coeff, train_accuracy, test_accuracy])
    
    # Convert the accuracy list to a pandas DataFrame and return it
    accuracy = pd.DataFrame(accuracy, columns=["regCoeff",'train accuracy in %', 'test accuracy in %']).set_index('regCoeff')
    return accuracy

### Fuction for Ploting Accuracy for each Regularisation Coefficience
def plot(df):
    """Create a bar graph using the input DataFrame and customize its appearance.

    Args:
        df: A pandas DataFrame containing accuracy values for different regularization coefficients.

    Returns:
        None. The function displays the graph using Matplotlib.

    Example:
        # Create a DataFrame
        df = pd.DataFrame({
            'Regularization Coefficients': [0.1, 1, 10],
            'Accuracy': [45, 50, 47]
        })

        # Call the plot function to create a bar graph
        plot(df)
    """
    # Plot a bar graph using the data from the input DataFrame
    df.plot(kind='bar')
    
    # Set the y-axis limit of the plot to be between 0 and 50
    plt.ylim(0, 50)
    
    # Set the title of the plot to be 'Accuracy for different regularization coefficients'
    plt.title('Accuracy for different regularization coefficients')
    
    # Set the label for the x-axis of the plot to be 'Regularization Coefficients'
    plt.xlabel('Regularization Coefficients')
    
    # Set the label for the y-axis of the plot to be 'Accuracy'
    plt.ylabel('Accuracy')
    
    # Display the plot
    return plt.show()

print("QUESTION FIVE\n")
### Regularised Multiclass for 1 vs rest
# Preparing binary training and testing data for Regularised Multiclass for 1 vs rest
reg_multi_class1vsrest_train = dataPrep(df_train, 1, multiclass=True)
reg_multi_class1vsrest_test = dataPrep(df_test, 1, multiclass=True)

print("Accuracy for Regularised Multiclass for 1 vs rest")
# print(f" Weight and Bias for 0.01 Regularised Multiclass for 1 vs rest: {RegperceptronTrain(Multi_class1vsrest_train, 20, 0.01)}")
reg_multi_class1vsrest_accuracy = RegAccuracy(reg_multi_class1vsrest_train, reg_multi_class1vsrest_test, 20, regCoeff)
print(f"{reg_multi_class1vsrest_accuracy}\n")
# print(plot(reg_multi_class1vsrest_accuracy))

### Regularised Multiclass for 2 vs rest
# Preparing binary training and testing data for Regularised Multiclass for 2 vs rest
reg_multi_class2vsrest_train = dataPrep(df_train, 2, multiclass=True)
reg_multi_class2vsrest_test = dataPrep(df_test, 2, multiclass=True)

print("Accuracy for Regularised Multiclass for 2 vs rest")
# print(f" Weight and Bias for 0.01 Regularised Multiclass for 2 vs rest: {RegperceptronTrain(reg_multi_class2vsrest_train, 20, 0.01)}")
reg_multi_class2vsrest_accuracy = RegAccuracy(reg_multi_class2vsrest_train, reg_multi_class2vsrest_test, 20, regCoeff)
print(f"{reg_multi_class2vsrest_accuracy}\n")
# print(plot(reg_multi_class2vsrest_accuracy))

### Regularised Multiclass for 3 vs rest
# Preparing binary training and testing data for Regularised Multiclass for 3 vs rest
reg_multi_class3vsrest_train = dataPrep(df_train, 3, multiclass=True)
reg_multi_class3vsrest_test = dataPrep(df_test, 3, multiclass=True)


print("Accuracy for Regularised Multiclass for 3 vs rest")
# print(f" Weight and Bias for 0.01 Regularised Multiclass for 2 vs rest: {RegperceptronTrain(reg_multi_class3vsrest_train, 20, 0.01)}")
reg_multi_class3vsrest_accuracy = RegAccuracy(reg_multi_class3vsrest_train, reg_multi_class3vsrest_test, 20, regCoeff)
print(f"{reg_multi_class3vsrest_accuracy}\n")
# print(plot(reg_multi_class2vsrest_accuracy))