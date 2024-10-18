# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Necessary Python Library And Read The CSV File Using Pandas
2. Conduct Necessary Preprocessing Steps 
3. Use Lable Encoder To Convert All The Datas Into Numberical Values
4. Split The Data Set For Training And Testing
5. Store The DecisionTreeClassifier With Entropy Criterion Inside A Variable
6. Fit the Model
7. Conduct Prediction, Evaluate Accuracy and Predict The Value On Test Data

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: VINOTH M P
RegisterNumber:  212223240182
*/
import pandas as pd
data=pd.read_csv( "Employee.csv" )
data . info( )
data. isnull() . sum()
data ["left"].value_counts ( )

from sklearn. preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
y=data["left"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test=train_test_split(x, y , test_size=0.2 ,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```


## Output:
![image](https://github.com/user-attachments/assets/ff26ad8d-64a1-47ea-8874-4019c4c942b4)

Null Values:

![image](https://github.com/user-attachments/assets/9b4f6ca5-4051-4270-a610-7ae3d06cdd72)

![image](https://github.com/user-attachments/assets/3d363f95-5841-4967-87c1-2a3733d005fd)

![image](https://github.com/user-attachments/assets/7195cdea-d502-41d0-87fc-f4deec8c87f6)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
