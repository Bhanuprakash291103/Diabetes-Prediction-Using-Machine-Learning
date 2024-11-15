import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc

df = pd.read_csv('updated_diabetes_data.csv')
df_copy = df.copy(deep=True)
df_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'WHR', 'Cholesterol']] = df_copy[
    ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'WHR', 'Cholesterol']].replace(0, np.nan)

df_copy['Glucose'] = df_copy['Glucose'].fillna(df_copy['Glucose'].mean())
df_copy['BloodPressure'] = df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].mean())
df_copy['SkinThickness'] = df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].median())
df_copy['Insulin'] = df_copy['Insulin'].fillna(df_copy['Insulin'].median())
df_copy['BMI'] = df_copy['BMI'].fillna(df_copy['BMI'].median())
df_copy['WHR'] = df_copy['WHR'].fillna(df_copy['WHR'].mean())
df_copy['Cholesterol'] = df_copy['Cholesterol'].fillna(df_copy['Cholesterol'].mean())

le = LabelEncoder()
df_copy['Physical_Activity_Level'] = le.fit_transform(df_copy['Physical_Activity_Level'])

X = df_copy.drop(columns='Outcome')
y = df_copy['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

rf_classifier = RandomForestClassifier(n_estimators=20, random_state=0)
svm_classifier = SVC(kernel='linear', probability=True, random_state=0)
dt_classifier = DecisionTreeClassifier(random_state=0)
gb_classifier = GradientBoostingClassifier(random_state=0)
knn_classifier = KNeighborsClassifier(n_neighbors=5)

rf_classifier.fit(X_train, y_train)
svm_classifier.fit(X_train, y_train)
dt_classifier.fit(X_train, y_train)
gb_classifier.fit(X_train, y_train)
knn_classifier.fit(X_train, y_train)

rf_y_pred = rf_classifier.predict(X_test)
svm_y_pred = svm_classifier.predict(X_test)
dt_y_pred = dt_classifier.predict(X_test)
gb_y_pred = gb_classifier.predict(X_test)
knn_y_pred = knn_classifier.predict(X_test)

print(f"Random Forest Accuracy: {round(accuracy_score(y_test, rf_y_pred) * 100, 2)}%")
print(f"SVM Accuracy: {round(accuracy_score(y_test, svm_y_pred) * 100, 2)}%")
print(f"Decision Tree Accuracy: {round(accuracy_score(y_test, dt_y_pred) * 100, 2)}%")
print(f"Gradient Boosting Accuracy: {round(accuracy_score(y_test, gb_y_pred) * 100, 2)}%")
print(f"KNN Accuracy: {round(accuracy_score(y_test, knn_y_pred) * 100, 2)}%")

plt.figure(figsize=(25, 10))
models = [('Random Forest', rf_y_pred), ('SVM', svm_y_pred), ('Decision Tree', dt_y_pred),
          ('Gradient Boosting', gb_y_pred), ('KNN', knn_y_pred)]
for i, (name, y_pred) in enumerate(models, start=1):
    plt.subplot(2, 3, i)
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='g', cmap='viridis')
    plt.title(f'Confusion Matrix for {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
plt.tight_layout()
plt.show()

feature_importances = pd.Series(rf_classifier.feature_importances_, index=X.columns)
plt.figure(figsize=(10, 6))
feature_importances.nlargest(10).plot(kind='barh', color='teal')
plt.title('Top 10 Important Features - Random Forest')
plt.xlabel('Feature Importance')
plt.show()

plt.figure(figsize=(15, 6))
sns.boxplot(data=df_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'WHR', 'Cholesterol']], orient='h')
plt.title('Distribution of Key Features')
plt.xlabel('Feature Value')
plt.show()

rf_y_pred_proba = rf_classifier.predict_proba(X_test)[:, 1]
svm_y_pred_proba = svm_classifier.predict_proba(X_test)[:, 1]
dt_y_pred_proba = dt_classifier.predict_proba(X_test)[:, 1]
gb_y_pred_proba = gb_classifier.predict_proba(X_test)[:, 1]
knn_y_pred_proba = knn_classifier.predict_proba(X_test)[:, 1]

plt.figure(figsize=(10, 7))
for model_name, y_pred_proba, color in [('Random /'
                                         'Forest', rf_y_pred_proba, 'blue'),
                                        ('SVM', svm_y_pred_proba, 'green'),
                                        ('Decision Tree', dt_y_pred_proba, 'purple'),
                                        ('Gradient Boosting', gb_y_pred_proba, 'orange'),
                                        ('KNN', knn_y_pred_proba, 'red')]:
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc(fpr, tpr):.2f})', color=color)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc="lower right")
plt.show()