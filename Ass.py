import numpy as np 

import pandas as pd 

from sklearn.linear_model import LogisticRegression 

from sklearn.metrics import( 

            accuracy_score, 
            precision_score, 
            recall_score, 
            f1_score, 
            roc_curve, 
            auc 
) 

import matplotlib.pyplot as plt 

 

## Data provided 

data = { 

          "Age":  [30, 45, 25], 

          "Salary":   [40000, 60000, 30000], 

           "Churn":   [1, 0,  1] 

} 

 

## convert to DataFrame 

df = pd.DataFrame(data) 

 

## separating independent X and dependent Y variables  

X = df[["Age", "Salary"]] 

y = df['Churn'] 

 

## creating and training the logistic regression model  

model = LogisticRegression() 

model.fit(X,  y) 

 

## predict probabilities and classes  

y_class_pred = model.predict(X) 

y_prob = model.predict_proba(X)[:, 1] 

 

## Evaluate the model performance  

accuracy = accuracy_score(y, y_class_pred) 
precision = precision_score(y, y_class_pred) 
recall = recall_score(y, y_class_pred) 
f1 = f1_score(y, y_class_pred) 

## print evaluation metrics
print("Model Performance:")
print(f"Accuracy:  {accuracy:.2f}")
print(f"Precision:  {precision:.2f}")
print(f"Recall:   {recall:.2f}")
print(f"F1 Score:    {f1:.2f}")

 

## plot the ROC curve and calculate the AUC score 

fpr, tpr, _ = roc_curve(y, y_prob) 

roc_auc = auc(fpr, tpr) 

 

plt.figure() 

plt.plot(fpr, tpr, color="darkorange",  lw=2, label=f"ROC curve  (area = {roc_auc:.2f})") 

plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")

plt.xlabel("False Positive Rate") 

plt.ylabel("True Positive Rate") 

plt.title("Receiver Operating Characteristic (ROC) Curve") 

plt.legend(loc="lower right")

plt.show() 

print(f"AUC Score:  {roc_auc:.2f}") 