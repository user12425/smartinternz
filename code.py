import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

csv_url = 'smoke_detection_iot.csv'

# csv_url = "C:\Users\sadiy\Downloads\smoke_detection_iot.csv"

df = pd.read_csv(csv_url)

print(df.head())
df.shape
df.info()

df.drop(columns= ['Unnamed:0','UTC'], axis=1, inplace=TRUE)

df.describe


plt.pie(df['Fire Alarm'].value_counts(), [0.2,0],labels=['Fire', 'No Fire'], autopct='%1.1f%%',colors=['red', 'black'])
plt.title('Fire Alarm')
plt.show()

sns.displot(df['Temperature[C]'])
sns.displot(df['Humidity[%]'])

df.drop(columns=['NC1.0', 'PM1.0'], axis = 1,inplace =True)
df.drop(columns=['NC2.5', 'PM12.5', 'eCO2[ppm]'], axis = 1,inplace =True)

X=df.drop(columns = ['Fire Alarm'])
y=df['Fire Alarm']

X = df.drop(columns=['Fire Alarm'])
y = df['Fire Alarm']
from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()
X_scaled = pd.DataFrame(scale.fit_transform(X), columns=X.columns) 
X_scaled.head()


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=0)


#smote
from imblearn.over_sampling import SMOTE
smote = SMOTE()
x_train_smote, y_train_smote =y_train_smote.value_counts()
smote.fit_resample(x_train, y_train)




## ------LOGISTIC REGRESSION MODEL

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

model_lr= LogisticRegression()
model_lr.fit(x_train_smote, y_train_smote)
y_pred_test_lr= model_lr.predict(x_test)
y_pred_train_lr= model_lr.predict(x_train_smote)
test_acc_lr= accuracy_score(y_test, y_pred_test_lr)
train_acc_lr= accuracy_score(y_train_smote, y_pred_train_lr)

print("Logistic Regression Test Accuracy:  ", test_acc_lr)
print(classification_report(y_text, y_pred_test_lr))


##-------- SVM MODEL

from sklearn.svm import SVC 
model_svm = SVC()
model_svm.fit(x_train_smote, y_train_smote) 
y_pred_test_svm = model_svm.predict(x_test)
y_pred_train_svm = model_svm.predict(x_train_smote)
test_acc_svm = accuracy_score (y_test, y_pred_test_svm) 
train_acc_svm = accuracy_score (y_train_smote, y_pred_train_svm) 
print('SVM Test Accuracy: ', test_acc_svm)
print(classification_report(y_test, y_pred_test_svm))


##---------- KNN MODEL

from sklearn.neighbors import KNeighborsClassifier 
model_knn =KNeighborsClassifier()
model_knn.fit(x_train_smote, y_train_smote) 
y_pred_test_knn = model_knn.predict(x_test) 
y_pred_train_knn = model_knn.predict(x_train_smote)
test_acc_knn = accuracy_score (y_test, y_pred_test_knn)
train_acc_knn= accuracy_score (y_train_smote, y_pred_train_knn)
print('KNN Test Accuracy: ', test_acc_knn)


##--------- TESTING MODELS
model_lr.predict([[20.05,55.28,0,12390,18849,939.736,0,3]])
model_svm.predict([[20.05,55.28,0,12390,18849,939.736,0,3]])
model_knn.predict([[20.05,55.28,0,12390,18849,939.736,0,3]])


#------------ COMPARING THE 3 MODELS

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
model_names = ['Logistic Regression', 'SVM', 'KNN']   # list of model names

y_pred_tests = [ y_pred_test_lr, y_pred_test_svm, y_pred_test_gb, y_pred_test_knn]    # list of predicted test labels for each model

results_df = pd.DataFrame(columns=['Model', 'Test Accuracy', 'Precision', 'Recall', 'F1-score'])

# calculating evaluation metrics for each model

for i, model_name in enumerate (model_names):
    model = model_names[i]
    y_pred_test = y_pred_tests[i]
    test_acc = accuracy_score (y_test, y_pred_test)
    classification = classification_report (y_test, y_pred_test, output_dict=True) 
    precision = classification['macro avg']['precision']
    recall = classification['macro avg']['recall'] 
    f1_score = classification['macro avg']['f1-score']
    results_df = results_df.append({ 'Model': model_name,
                                'Test Accuracy': test_acc, 
                                'Precision': precision, 
                                'Recall': recall,
                                'F1-score': f1_score}, ignore_index=True)

# results_df shows comparison as a table


#--- SAVING BEST MODEL IN PKL -----#
import pickle
with open('smoke.pkl', 'wb') as file: 
    pickle.dump (model_lr, file)


prediction = model.predict(final_features)[0]


from flask import Flask, render_template, request
with open('smoke.pkl', 'rb') as f:
    model = pickle.load(f)




"""@app.route('/')
def home():
input_data = df
prediction = model.predict(input_data)

if prediction == 1:
    smoke_detected = "Yes"
else:
    smoke_detected = "No"

return render_template('predict.html', smoke_detected=smoke_detected)    

    # Pass the prediction result to the template
    return render_template('predict.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)"""






from flask import Flask, render_template, request
app= Flask(__name__) #server-app interface

model = pickle.load(open ('smoke.pkl', 'rb'))
@app.route('/') #binds to url
def helloworld():
    return render_template("index.html")
@app.route('/smoke_detection', methods =['POST']) #binds to url
def smoke_detection(): 
    p =request.form['ut'] 
    q= request.form['te'] 
    r= request.form['hu']
    s= request.form['tv']
    t=request.form['co']
    u=request.form['ra']
    v=request.form['et']
    w=request.form['pr']
    x=request.form['pm']
    y=request.form['pn']
    z=request.form['nc']
    a=request.form['nd']
    b=request.form['ne']
    c=request.form['cn']
total=np.array([[p,q,r,s,t,u,v,w,x,y,za,b,c,d]])
output= model.predict(total)
print(output)


if __name__ == '__main__':
    app.run(debug=False)


#-------- RENDER HTML PAGE ----------#

def home():
    return render_template('home.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/submit', methods=['POST'])
def submit():
    temperature =float(request.form['Temperature[C]']) 
    humidity = float(request.form['Humidity[%]']) 
    tvoc = float(request.form['TVOC[ppb]'])
    raw_h2 = float(request.form['Raw H2'])
    raw_ethanol =float(request.form['Raw Ethanol']) 
    pressure = float(request.form['Pressure [hPa]']) 
    nce_5 =float(request.form['NCO.5'])
    cnt = float(request.form['CNT'])
    
    final_features = np.array([[temperature, humidity, tvoc, raw_h2, raw_ethanol, pressure, cnt, file_alarm]])
    
    # making the prediction
    prediction = model.predict(final_features)[0]
