# PRODIGY_DS_02
📋 Project Overview
This repository contains the second task of my Prodigy InfoTech internship. The goal was to perform comprehensive Data Cleaning and Exploratory Data Analysis (EDA) on a dataset to uncover patterns, trends, and relationships between variables.

📊 Dataset
I used the Titanic - Machine Learning from Disaster dataset, which provides information on the passengers aboard the Titanic, including whether they survived or not.

Dataset Link: Titanic Dataset (Prodigy Task 2)

🔍 Workflow
The analysis was divided into three main phases:

1. Data Cleaning
Handling missing values in the 'Age', 'Cabin', and 'Embarked' columns.

Removing duplicates and correcting data types.

Feature engineering (e.g., creating a 'FamilySize' variable from 'SibSp' and 'Parch').

2. Exploratory Data Analysis (EDA)
Univariate Analysis: Examining the distribution of survival rates, gender, and passenger classes.

Bivariate Analysis: Investigating the relationship between survival and age, or survival and fare price.

Multivariate Analysis: Using heatmaps to find correlations between all numerical features.

🛠️ Tech Stack
Python

Pandas (Cleaning & Wrangling)

Seaborn & Matplotlib (Statistical Visualization)

NumPy (Numerical Operations)

💡 Key Insights
Gender Bias: Female passengers had a significantly higher survival rate than male passengers.

Socio-economic Factor: Passengers in First Class (Pclass 1) were more likely to survive than those in Third Class.

Age Factor: Children had a higher survival probability, appearing as a distinct peak in the age distribution of survivors.
