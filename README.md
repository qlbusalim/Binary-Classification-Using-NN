# Artificial Neural Network Project - Backpropagation

This project demonstrates the use of a neural network with backpropagation to predict credit card default. The model is built using TensorFlow and Keras, and the dataset used is the "Default of Credit Card Clients" dataset from the UCI Machine Learning Repository.

## Libraries Used

* pandas
* numpy
* matplotlib
* seaborn
* ucimlrepo
* scikit-learn
* imbalanced-learn
* scipy
* tensorflow
* keras

## Workflow

1.  **Data Loading and Exploration**: The "Default of Credit Card Clients" dataset is fetched from the UCI repository. The dataset contains 23 features and a binary target variable indicating default payment.

2.  **Data Cleaning**: Invalid values in the 'X3' (Education) and 'X4' (Marital Status) columns are identified and replaced with NaN.

3.  **Exploratory Data Analysis (EDA)**: The distributions of numerical and categorical features are visualized using histograms and count plots to understand the data's characteristics.

4.  **Data Preprocessing**:
    * **Missing Value Imputation**: Missing values are imputed using the most frequent value for each respective column.
    * **Outlier Handling**: Outliers in the numerical features are handled by winsorizing.
    * **Train-Test Split**: The data is split into training and testing sets.
    * **Feature Scaling**: Numerical features are scaled using `StandardScaler`.
    * **Handling Class Imbalance**: The `SMOTEENN` technique is applied to the training data to address the class imbalance between default and non-default clients.

5.  **Model Building**: A sequential neural network model is constructed using Keras with the following architecture:
    * An input layer.
    * Three hidden layers with ReLU activation.
    * A dropout layer to prevent overfitting.
    * An output layer with sigmoid activation for binary classification.

6.  **Model Training and Evaluation**:
    * The model is compiled with the Adam optimizer and binary cross-entropy loss function.
    * Early stopping and model checkpoint callbacks are used during training to save the best model and prevent overfitting.
    * The model is trained on the preprocessed training data.
    * The trained model is evaluated on the test set, and a classification report, confusion matrix, and other metrics like precision, recall, and F1-score are generated.

## Results

The model's performance on the test set is as follows:

* **Precision**: [_Fill in from notebook output_]
* **Recall**: [Fill in from notebook output]
* **F1-Score**: [Fill in from notebook output]

## How to Run

1.  Make sure you have all the required libraries installed. You can install them using pip:
    ```bash
    pip install pandas numpy matplotlib seaborn ucimlrepo scikit-learn imbalanced-learn scipy tensorflow keras
    ```
2.  Run the Jupyter Notebook `Backpropagation.ipynb`.
