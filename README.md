## Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

### AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

### Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

### Algorithm

Step 1: Read the employee data from a CSV file.

Step 2: Check for null values and encode categorical variables.

Step 3: Define the features (X) and target (y).

Step 4: Split the data into training and testing sets.

Step 5: Train the Decision Tree Classifier.

step 6: Make predictions on the test data.

step 7: Calculate the accuracy of the model.

step 8: Use the model to predict new data.

### Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: VINOTH M P M P
RegisterNumber:  21222324010182
*/
import pandas as pd
data=pd.read_csv( "Employee.csv" )
data . info( )
data.isnull().sum()
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
print("Predict:",dt.predict([[0.5,0.8,9,260,6,0,1,2]]))
```
### Output:

![image](https://github.com/user-attachments/assets/ff26ad8d-64a1-47ea-8874-4019c4c942b4)

![image](https://github.com/user-attachments/assets/a62fa533-1e1e-4586-b065-d64883520808)

![image](https://github.com/user-attachments/assets/3d363f95-5841-4967-87c1-2a3733d005fd)

![image](https://github.com/user-attachments/assets/54371204-5cf3-4555-b59a-ce92fa2c4e2b)

### Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
