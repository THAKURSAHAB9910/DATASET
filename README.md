# DATASET
COVID-19 MORTALITY ANALYSIS
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# 1. Load Dataset
file_path = "/mnt/data/Provisional_COVID-19_Deaths_by_Sex_and_Age (1).csv"
df = pd.read_csv("Provisional_COVID-19_Deaths_by_Sex_and_Age (1).csv")

# 2. Dataset Overview
print("Dataset Info:\n")
print(df.info())

# 3. Check Missing Values
print("\nMissing Values:\n", df.isnull().sum())
print("\nTotal Missing Values:\n", df.isnull().sum().sum())

# 4. Handle Missing Values
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].median())

categorical_columns = df.select_dtypes(include=['object']).columns
df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])

# 5. Summary Statistics
print("\nSummary Statistics:\n", df.describe())

# 6. Histograms of Numerical Features
numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_features].hist(bins=30, figsize=(12, 8), color='skyblue', edgecolor='black')
plt.suptitle("Histograms of Numerical Features")
plt.tight_layout()
plt.show()

# 7. Boxplots to Detect Outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[numerical_features])
plt.xticks(rotation=45)
plt.title("Boxplot of Numerical Features")
plt.tight_layout()
plt.show()

# 8. Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# 9. Outlier Detection
# IQR Method
Q1 = df[numerical_features].quantile(0.25)
Q3 = df[numerical_features].quantile(0.75)
IQR = Q3 - Q1
outliers_iqr = ((df[numerical_features] < (Q1 - 1.5 * IQR)) | (df[numerical_features] > (Q3 + 1.5 * IQR))).sum()
print("\nOutliers Detected (IQR Method):\n", outliers_iqr)

# Z-score Method
z_scores = np.abs(stats.zscore(df[numerical_features]))
outliers_zscore = (z_scores > 3).sum()
print("\nOutliers Detected (Z-score Method):\n", outliers_zscore)

# 10. Deaths by Sex and Age Group
if 'Sex' in df.columns and 'Age Group' in df.columns:
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='Age Group', hue='Sex')
    plt.title("Distribution of Deaths by Sex and Age Group")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 11. Top Areas/States by Total Deaths
if 'COVID-19 Deaths' in df.columns and 'State' in df.columns:
    top_states = df.groupby('State')['COVID-19 Deaths'].sum().sort_values(ascending=False).head(10)
    top_states.plot(kind='barh', color='coral')
    plt.title("Top 10 States by COVID-19 Deaths")
    plt.xlabel("Total Deaths")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# 12. Time Trend (if time info available)
date_col = [col for col in df.columns if 'week' in col.lower() or 'date' in col.lower()]
if date_col:
    df[date_col[0]] = pd.to_datetime(df[date_col[0]], errors='coerce')
    time_df = df.groupby(date_col[0])['COVID-19 Deaths'].sum()
    time_df.plot(figsize=(12, 5), title="COVID-19 Deaths Over Time", color='green')
    plt.ylabel("Total Deaths")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.show()

# 13. Grouped Mean Deaths by Sex and Age
if 'Sex' in df.columns and 'COVID-19 Deaths' in df.columns:
    grouped = df.groupby('Sex')['COVID-19 Deaths'].mean().sort_values()
    grouped.plot(kind='bar', color='teal', title="Average COVID-19 Deaths by Sex")
    plt.ylabel("Average Deaths")
    plt.tight_layout()
    plt.show()

if 'Age Group' in df.columns and 'COVID-19 Deaths' in df.columns:
    grouped = df.groupby('Age Group')['COVID-19 Deaths'].mean().sort_values()
    grouped.plot(kind='barh', color='purple', title="Average COVID-19 Deaths by Age Group")
    plt.xlabel("Average Deaths")
    plt.tight_layout()
    plt.show()





# 15. Pie Chart - Death Distribution by Sex
if 'Sex' in df.columns and 'COVID-19 Deaths' in df.columns:
    death_by_sex = df.groupby('Sex')['COVID-19 Deaths'].sum()
    plt.figure(figsize=(6, 6))
    death_by_sex.plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff','#99ff99'])
    plt.title("COVID-19 Death Distribution by Sex")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()

# 16. KDE Plot - Distribution of COVID-19 Deaths
if 'COVID-19 Deaths' in df.columns:
    plt.figure(figsize=(8, 5))
    sns.kdeplot(df['COVID-19 Deaths'], fill=True, color='navy')
    plt.title("KDE Plot of COVID-19 Deaths")
    plt.tight_layout()
    plt.show()


# 19. Scatterplot - Check relation between any two numeric features
if len(numerical_features) >= 2:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=numerical_features[0], y=numerical_features[1])
    plt.title(f"Scatterplot: {numerical_features[0]} vs {numerical_features[1]}")
    plt.tight_layout()
    plt.show()



