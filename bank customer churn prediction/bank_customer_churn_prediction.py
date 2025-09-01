import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import zipfile
import os
filepath = '/content/bank-customer-churn-prediction'
with zipfile.ZipFile(filepath, 'r') as zip_ref:
    zip_ref.extractall('/content')

import os
print(os.listdir('/content'))

df = pd.read_csv('/content/Churn_Modelling.csv')
display(df.head())

display(df.head())
df.info()
display(df.describe())

display(df.isnull().sum())

plt.figure(figsize=(10, 5))
sns.countplot(x='Exited', data=df)
plt.title('Distribution of Exited Customers')
plt.xlabel('Exited')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10, 5))
sns.histplot(df['CreditScore'], kde=True)
plt.title('Distribution of CreditScore')
plt.xlabel('CreditScore')
plt.ylabel('Frequency')
plt.show()

numerical_cols = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(x='Exited', y=col, data=df)
    plt.title(f'Distribution of {col} by Exited')
plt.tight_layout()
plt.show()

categorical_cols = ['Geography', 'Gender', 'NumOfProducts', 'HasCrCard', 'IsActiveMember']
plt.figure(figsize=(15, 15))
for i, col in enumerate(categorical_cols):
    plt.subplot(3, 2, i + 1)
    sns.countplot(x=col, hue='Exited', data=df)
    plt.title(f'Distribution of {col} by Exited')
plt.tight_layout()
plt.show()

numerical_df = df[numerical_cols]
correlation_matrix = numerical_df.corr()
display(correlation_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.show()

for col in numerical_cols:
    df[col] = df[col].fillna(df[col].mean())

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

display(df.isnull().sum())

df = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols)

X = df.drop('Exited', axis=1)
y = df['Exited']

display(X.head())
display(y.head())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

X_train = X_train.drop(['Surname', 'RowNumber', 'CustomerId'], axis=1)
X_test = X_test.drop(['Surname', 'RowNumber', 'CustomerId'], axis=1)

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

y_pred = xgb_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()