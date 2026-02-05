import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("titanic.csv")

# Display basic info
print(data.head())
print(data.info())
print(data.describe())

# -----------------------------
# Data Cleaning
# -----------------------------

# Fill missing Age with median
data['Age'].fillna(data['Age'].median(), inplace=True)

# Fill missing Embarked with mode
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Drop Cabin column (too many missing values)
data.drop(columns=['Cabin'], inplace=True)

# -----------------------------
# Exploratory Data Analysis
# -----------------------------

# Survival count
plt.figure()
sns.countplot(x='Survived', data=data)
plt.title("Survival Count")
plt.show()

# Survival by Gender
plt.figure()
sns.countplot(x='Sex', hue='Survived', data=data)
plt.title("Survival by Gender")
plt.show()

# Age distribution
plt.figure()
sns.histplot(data['Age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()

# Survival by Passenger Class
plt.figure()
sns.countplot(x='Pclass', hue='Survived', data=data)
plt.title("Survival by Passenger Class")
plt.show()

# Correlation heatmap
plt.figure()
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()