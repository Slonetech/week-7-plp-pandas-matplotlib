
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Set visual style
sns.set(style="whitegrid")

# Task 1: Load and Explore the Dataset
try:
    # Load iris dataset from sklearn and convert to DataFrame
    iris_data = load_iris()
    df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
    df['species'] = pd.Categorical.from_codes(iris_data.target, iris_data.target_names)
    
    # Display first few rows
    print(" First 5 rows of the dataset:")
    print(df.head())

    # Data types and missing values
    print("\n Data Types:")
    print(df.dtypes)

    print("\n Missing Values:")
    print(df.isnull().sum())

except FileNotFoundError as e:
    print(" Dataset file not found:", e)
except Exception as e:
    print(" Error loading the dataset:", e)

# Task 2: Basic Data Analysis
print("\n📈 Descriptive Statistics:")
print(df.describe())

# Group by species and compute mean of numerical columns
grouped = df.groupby('species').mean()
print("\n Mean values per species:")
print(grouped)

# Observations
print("\n🔍 Observations:")
print("Setosa has the smallest petals and sepals; Virginica the largest on average.")

# Task 3: Data Visualization

# Line Chart - Simulated trend (not a real time series)
plt.figure(figsize=(8, 4))
plt.plot(df.index, df['sepal length (cm)'], label='Sepal Length')
plt.title("Line Chart: Sepal Length Over Index")
plt.xlabel("Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.tight_layout()
plt.show()

# Bar Chart - Average petal length per species
plt.figure(figsize=(6, 4))
sns.barplot(x=grouped.index, y=grouped['petal length (cm)'])
plt.title("Bar Chart: Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
plt.tight_layout()
plt.show()

# Histogram - Distribution of sepal width
plt.figure(figsize=(6, 4))
plt.hist(df['sepal width (cm)'], bins=15, color='skyblue', edgecolor='black')
plt.title("Histogram: Sepal Width Distribution")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Scatter Plot - Sepal vs Petal length
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species')
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title='Species')
plt.tight_layout()
plt.show()
