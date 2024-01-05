# Deep-Learning-Challenge
## Alphabet Soup Charity Analysis
This repository contains a deep learning model created to predict which organizations are most likely to receive funding from Alphabet Soup Charity. The model was developed using Python and the TensorFlow library, and trained on a dataset containing over 34,000 organizations that have previously received funding.

## Getting Started
To run the model, you will need to have the following software installed on your machine:

Python 3
TensorFlow 2
Pandas
NumPy
Scikit-learn
You can install these packages using pip or conda. Once you have the necessary software installed, you can clone this repository to your local machine and run the AlphabetSoupCharity.ipynb notebook to train and evaluate the model.

## Dataset
The dataset used in this analysis is included in the repository (charity_data.csv). It contains the following columns:

EIN: the organization's Employer Identification Number
Name: the name of the organization
Application_Type: the type of application the organization submitted to receive funding
Affiliation: whether the organization is independent or affiliated with a larger group
Classification: the organization's classification as a charity or foundation
Use_Case: the organization's proposed use for the funding
Organization: the type of organization (e.g. corporation, trust, association)
Status: the organization's IRS filing status
Income_Amt: the organization's reported income
Special_Considerations: whether the organization requires special considerations for funding
Ask_Amt: the amount of funding requested
Is_Successful: whether the organization received funding from Alphabet Soup Charity (the target variable)
Model
The deep learning model used in this analysis is a sequential neural network with three hidden layers. The input layer has 51 nodes (equal to the number of features in the dataset), and the output layer has one node (for the binary classification task). The first hidden layer has 80 nodes, the second has 30 nodes, and the third has 10 nodes. All hidden layers use the ReLU activation function, and the output layer uses the sigmoid activation function.

The model was compiled using the binary cross-entropy loss function and the Adam optimizer, and trained for 100 epochs with a batch size of 1000. A checkpoint was created to save the weights with the best validation accuracy, and early stopping was used to prevent overfitting.

## Results
After training the model on the dataset, we achieved an accuracy of 76.4% on the test set. While this is above our target accuracy of 75%, there is still room for improvement.

To increase the model's performance, we could try the following:

Collect more data: With more data, the model may be able to identify more patterns and make more accurate predictions.
Feature engineering: We could create new features based on the existing variables to help the model better capture the relationships between the variables.
Hyperparameter tuning: We could try different numbers of neurons, layers, and activation functions to see if we can improve the model's performance.
Conclusion
Overall, the deep learning model created in this analysis shows promise for predicting which organizations are most likely to receive funding from Alphabet Soup Charity. With further optimization and tuning, we may be able to achieve even better accuracy and help Alphabet Soup Charity make more informed decisions about where to allocate their resources.