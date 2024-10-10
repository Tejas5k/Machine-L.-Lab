import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

try:
    print("Dataset: \n")
    dataset = pd.read_csv(r"D:\ML\lab\Assignment-5\Emails.csv")  
except FileNotFoundError:
    print("Error: File 'Emails.csv' not found. Please check the path.")
    exit()

dataset['Prediction'] = dataset['Prediction'].replace(0, 'spam')
dataset['Prediction'] = dataset['Prediction'].replace(1, 'ham')

print(dataset)

x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

xtr, xt, ytr, yt = train_test_split(x, y, test_size=0.2, random_state=90)

print("\nTraining Dataset Independent Variable:\n")
print(xtr)
print("\nTraining Dataset Dependent Variable:\n")
print(ytr)
print("\nTesting Dataset Independent Variable:\n")
print(xt)
print("\nTesting Dataset Dependent Variable:\n")
print(yt)

sc = StandardScaler()
xtr = sc.fit_transform(xtr)
xt = sc.transform(xt)

print("\nAfter Standardizing Dataset :\n")
print(xtr)

svm = SVC(kernel='linear', random_state=0)
svm.fit(xtr, ytr)

y_pred2 = svm.predict(xt)

print("\n\n\t\t\tClassification Report (SVM)")
print(classification_report(yt, y_pred2))

