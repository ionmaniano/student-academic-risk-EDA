
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import graphviz

# 1. Create the dataset
raw_data = {
    'Student_ID': [1001, 1002, 1003, 1004, 1005, 1001, 1006, 1007, 1008, 1009],
    'County': ['Nakuru', 'Narok', 'Nakuru', 'Narok', 'Kakamega', 'Nakuru', 'Nairobi', None, 'Narok', 'Nakuru'],
    'Study_Hours_Per_Week': [15, 8, 20, None, 12, 15, 30, 10, 25, 18],
    'Attendance_Percent': [85, 60, 92, 55, 78, 85, 95, 65, 88, None],
    'Fee_Balance_Ksh': [0, 15000, 0, 25000, 5000, 0, 0, 18000, 2000, 12000],
    'Previous_Grade': [72, 48, 85, 45, 65, 72, 90, 50, 82, 58],
    'Student_Type': ['Regular', 'Regular', 'Scholarship', 'Regular', 'Scholarship', 'Regular', 'Regular', 'Scholarship', 'Regular', 'Regular'],
    'Academic_Risk': [0, 1, 0, 1, 0, 0, 0, 1, 0, 1]
}

df = pd.DataFrame(raw_data)

print("\n--- 1. DATASET OVERVIEW ---")
print("Shape:", df.shape)
print("\nData types:\n", df.dtypes)
print("\nSummary statistics:\n", df.describe(include='all'))

# 2. Missing values and duplicates
print("\n--- 2. DATA QUALITY ---")
print("Missing values:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())
print("Duplicate Student_IDs:", df['Student_ID'].duplicated().sum())

# 3. Clean data
df['County'] = df['County'].fillna(df['County'].mode()[0])
df['Study_Hours_Per_Week'] = df['Study_Hours_Per_Week'].fillna(
    df['Study_Hours_Per_Week'].median()
)
df['Attendance_Percent'] = df['Attendance_Percent'].fillna(
    df['Attendance_Percent'].median()
)

# Remove Student_ID because it is an identifier, not a useful predictor
df = df.drop_duplicates()
df = df.drop(columns=['Student_ID'])

# 4. Feature engineering
df['Attendance_Category'] = pd.cut(
    df['Attendance_Percent'],
    bins=[-1, 59, 79, 100],
    labels=['Low', 'Medium', 'High']
)
df['Has_Fee_Balance'] = (df['Fee_Balance_Ksh'] > 0).astype(int)

print("\n--- 3. CLEANED DATA ---")
print(df.head(10))
print("\nMissing values after cleaning:\n", df.isnull().sum())

# 5. Exploratory data analysis plots
sns.set_theme(style="whitegrid")

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Academic_Risk')
plt.title('Number of Students by Academic Risk')
plt.savefig('academic_risk_count.png', bbox_inches='tight')
plt.close()

plt.figure(figsize=(7, 4))
sns.histplot(data=df, x='Study_Hours_Per_Week', hue='Academic_Risk',
             bins=6, multiple='stack')
plt.title('Study Hours and Academic Risk')
plt.savefig('study_hours_risk.png', bbox_inches='tight')
plt.close()

plt.figure(figsize=(7, 4))
sns.boxplot(data=df, x='Academic_Risk', y='Attendance_Percent')
plt.title('Attendance by Academic Risk')
plt.savefig('attendance_risk.png', bbox_inches='tight')
plt.close()

plt.figure(figsize=(7, 4))
sns.countplot(data=df, x='Has_Fee_Balance', hue='Academic_Risk')
plt.title('Fee Balance Indicator and Academic Risk')
plt.savefig('fee_balance_risk.png', bbox_inches='tight')
plt.close()

# 6. Encode categorical variables
model_df = df.copy()
model_df['County'] = LabelEncoder().fit_transform(model_df['County'])
model_df['Student_Type'] = LabelEncoder().fit_transform(model_df['Student_Type'])
model_df['Attendance_Category'] = LabelEncoder().fit_transform(
    model_df['Attendance_Category'].astype(str)
)

# 7. Correlation heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(model_df.corr(numeric_only=True), annot=True, cmap='coolwarm',
            fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', bbox_inches='tight')
plt.close()

# 8. Scale numerical predictors
feature_columns = [
    'County', 'Study_Hours_Per_Week', 'Attendance_Percent',
    'Fee_Balance_Ksh', 'Previous_Grade', 'Student_Type',
    'Attendance_Category', 'Has_Fee_Balance'
]
X = model_df[feature_columns]
y = model_df['Academic_Risk']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 9. Train a decision tree with maximum depth 3
# The dataset is very small, so results are illustrative, not reliable predictions.
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_scaled, y)

print("\n--- 4. DECISION TREE RESULTS ---")
print("Training accuracy:", round(accuracy_score(y, model.predict(X_scaled)), 3))
print("\nClassification report:\n",
      classification_report(y, model.predict(X_scaled), zero_division=0))

# 10. Export tree to Graphviz DOT and PNG
dot_data = export_graphviz(
    model,
    out_file=None,
    feature_names=feature_columns,
    class_names=['Not at Risk', 'At Risk'],
    filled=True,
    rounded=True,
    special_characters=True
)

with open('student_risk_tree.dot', 'w') as f:
    f.write(dot_data)

graph = graphviz.Source(dot_data)
graph.render('student_risk_tree', format='png', cleanup=True)

print("\nTree files created: student_risk_tree.dot and student_risk_tree.png")
print("EDA plots saved as PNG files in the current folder.")

# 11. Display decision rules
from sklearn.tree import export_text
print("\n--- 5. DECISION RULES ---")
print(export_text(model, feature_names=feature_columns))

print("\n--- ROOT FEATURE ---")
print("The root feature is:", feature_columns[model.tree_.feature[0]])











import pandas as pd
import numpy as np

# 1. Create the dataset
raw_data = {
    'Student_ID': [1001, 1002, 1003, 1004, 1005, 1001, 1006, 1007, 1008, 1009],
    'County': ['Nakuru', 'Narok', 'Nakuru', 'Narok', 'Kakamega',
               'Nakuru', 'Nairobi', None, 'Narok', 'Nakuru'],
    'Study_Hours_Per_Week': [15, 8, 20, None, 12, 15, 30, 10, 25, 18],
    'Attendance_Percent': [85, 60, 92, 55, 78, 85, 95, 65, 88, None],
    'Fee_Balance_Ksh': [0, 15000, 0, 25000, 5000, 0, 0, 18000, 2000, 12000],
    'Previous_Grade': [72, 48, 85, 45, 65, 72, 90, 50, 82, 58],
    'Student_Type': ['Regular', 'Regular', 'Scholarship', 'Regular',
                     'Scholarship', 'Regular', 'Regular', 'Scholarship',
                     'Regular', 'Regular'],
    'Academic_Risk': [0, 1, 0, 1, 0, 0, 0, 1, 0, 1]
}

df = pd.DataFrame(raw_data)

# 2. Inspect the dataset
print("\n--- DATASET ---")
print(df)

print("\n--- DIMENSIONS ---")
print(df.shape)

print("\n--- DATA TYPES ---")
print(df.dtypes)

print("\n--- SUMMARY STATISTICS ---")
print(df.describe(include='all'))

# 3. Identify duplicates and missing values
print("\n--- DUPLICATE ROWS ---")
print(df.duplicated().sum())

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

# 4. Clean the dataset
df = df.drop_duplicates().copy()

df['Study_Hours_Per_Week'] = df['Study_Hours_Per_Week'].fillna(
    df['Study_Hours_Per_Week'].median()
)
df['Attendance_Percent'] = df['Attendance_Percent'].fillna(
    df['Attendance_Percent'].median()
)
df['County'] = df['County'].fillna(df['County'].mode()[0])

print("\n--- CLEANED DATA ---")
print(df)

print("\n--- MISSING VALUES AFTER CLEANING ---")
print(df.isnull().sum())

print("\n--- CLEANED DIMENSIONS ---")
print(df.shape)

