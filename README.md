Titanic Survival Prediction This project aims to predict the survival of passengers on the Titanic using machine learning techniques. The dataset used in this project is the Titanic dataset, which contains information about the passengers, including their age, gender, class, and whether they survived or not.

Table of Contents Project Overview Dataset Installation Exploratory Data Analysis Preprocessing Model Training Prediction Conclusion Running the Project Project Overview The goal of this project is to predict whether a passenger survived the Titanic disaster based on various features such as their age, gender, and class. We will use a Logistic Regression model to make these predictions.

Dataset The dataset used in this project is the Titanic dataset. It can be downloaded from Kaggle. The dataset contains the following columns:

PassengerId: Unique ID for each passenger Survived: Survival indicator (0 = No, 1 = Yes) Pclass: Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd) Name: Passenger name Sex: Passenger gender Age: Passenger age SibSp: Number of siblings/spouses aboard Parch: Number of parents/children aboard Ticket: Ticket number Fare: Passenger fare Cabin: Cabin number Embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) Installation To run this project, you need to have Python installed along with the following libraries:

NumPy Pandas Matplotlib Seaborn Scikit-learn You can install these libraries using pip:

bash Copy code pip install numpy pandas matplotlib seaborn scikit-learn Exploratory Data Analysis We begin by loading the dataset and exploring the data to understand its structure and contents. This involves displaying the first few rows, checking for missing values, and visualizing the distribution of key features.

python Copy code import numpy as np import pandas as pd import matplotlib.pyplot as plt import seaborn as sns

df = pd.read_csv("Titanic-Dataset.csv") print(df.head()) print(df.shape) print(df.describe()) print(df['Survived'].value_counts())
