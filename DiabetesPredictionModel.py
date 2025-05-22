import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


diabetes_dataset=pd.read_csv('/home/ishaan/tensorflow_project/kaggle/diabetes.csv')
X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']

#Data Standardization

scaler = StandardScaler()
scaler.fit(X)
StandardScaler(copy=True, with_mean=True, with_std=True)
standardized_data = scaler.transform(X)
#print(standardized_data)

X = standardized_data
Y = diabetes_dataset['Outcome']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)


#training the model

classifier=svm.SVC(kernel='linear')

#training the support vector Machine Classifier


classifier.fit(X_train, Y_train)

#Model Evaluation

#Accuracy Score

X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
#print('Accuracy score of the training data : ', training_data_accuracy)

# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
#print('Accuracy score of the test data : ', test_data_accuracy)

#Making a Predictive System
preg=int(input('No of pregnancies you had;'))
glu=int(input('Whats your Glucose;'))
skthick=float(input('enter your skin thickness;'))
insulin=float(input('enter your insulin;'))
bp=float(input('enter your BP;'))
bmi=float(input('enter your BMI;'))
diabetespdf=float(input('enter your diabetes pedigree function;'))
age=float(input('enter your age;'))

input_data = (preg,glu,bp,skthick,insulin,bmi,diabetespdf,age)



# changing the input_data to numpy array
inputdata_asnumpyarray = np.asarray(input_data)

# reshape the array as we are predicting for one instance
inputdatareshaped = inputdata_asnumpyarray.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(inputdatareshaped)
#print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')