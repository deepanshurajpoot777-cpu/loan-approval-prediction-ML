
# ==========================
# STEP 1 : Load Dataset
# ==========================


import pandas as pd
df=pd.read_csv("loan_data.csv")

print("Data set Rows")
print(df.head())

print("\n Dataset")
print(df.shape)

print("\n Dataset info")
print(df.info())

print("\nSummary Statics")
print(df.describe())

print("\n MIssing Values")
print(df.isnull().sum())

print(df.columns)

# ==========================
# STEP 2 : Data Cleaning
# ==========================


from sklearn.preprocessing  import LabelEncoder

le=LabelEncoder()
df["person_gender"]=le.fit_transform(df["person_gender"]) # yes=1 no=0
df["previous_loan_defaults_on_file"]=le.fit_transform(df["previous_loan_defaults_on_file"])

df=pd.get_dummies(df,columns=["person_education"],drop_first=True,dtype=int)
df=pd.get_dummies(df,columns=["person_home_ownership"],drop_first=True,dtype=int)
df=pd.get_dummies(df,columns=["loan_intent"],drop_first=True,dtype=int)

print("\n After encodding")
print(df.tail())

print("\n Data Type")
print(df.dtypes)

print("\n After encoding")
print(df.columns)

#===========================
# STEP 3 : Model Training
# ==========================


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score

# Feature and Target
x=df.drop("loan_status",axis=1)
y=df["loan_status"]

# train test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# Feature scalling

scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

# model Training

model=LogisticRegression()
model.fit(x_train,y_train)

#prediction

y_pred=model.predict(x_test)
print("\nACtual loan_status")
print(y_test.head())

print("\nPredict")
print(y_pred[:5])

# Evaluation

print("\n Classification Report")
print(classification_report(y_test,y_pred))

print("Accuracy:",accuracy_score(y_test,y_pred))

# confusion metrics
conf_metrics=confusion_matrix(y_test,y_pred)

# Visulization

plt.figure(figsize=(8,6))
sns.heatmap(
    conf_metrics,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Rejected","Approved"],
    yticklabels=["Rejected","Approved"]
)
plt.xlabel("Predicted")
plt.ylabel("ACtual")
plt.title("Confusion Matrix")
plt.savefig("Confusion_matrix.png")
plt.show()


# ==========================
# STEP 4 : User Prediction
# ==========================


print("\npredict loan_status")

age=int(input("enter your Age:"))
gender=input("enter your gender:")
income=int(input("enter your income:"))
exp=int(input("enter your person_emp_exp :"))
loan=int(input("enter your loan Ammount:"))
rate=float(input("enter your loan_int_rate:"))
percent_income=float(input("enter your loan_percent_income:"))
cred_hist_length=int(input("enter your cb_person_cred_hist_length:"))
credit_score=int(input("enter your credit score:"))
previous_loan =input("enter your previous_loan_defaults_on_file(yes/no):")
education=input("enter your education:")
home_ownership=input("enter your person_home_ownership:")
intent=input("enter your lone_intant:")

user_input=pd.DataFrame(0,index=[0],columns=x.columns)

user_input["person_age"]=age
user_input["person_income"]=income
user_input["person_emp_exp"]=exp
user_input["loan_amnt"]=loan
user_input["loan_int_rate"]=rate
user_input["loan_percent_income"]=percent_income
user_input["cb_person_cred_hist_length"]=cred_hist_length
user_input["credit_score"]=credit_score

# encoding

user_input["person_gender"]=1 if gender.lower()=="male" else 0
user_input["previous_loan_defaults_on_file"]=1 if previous_loan.lower()=="yes" else 0

# education encodding

edu_col="person_education_"+education

if edu_col in user_input.columns:
    user_input[edu_col]=1

# person_home_ownership encodding

home_col="person_home_ownership_"+home_ownership

if home_col in user_input.columns:
    user_input[home_col]=1

#loan_intent encodding

loan_col ="loan_intent_"+intent

if loan_col in user_input.columns:
    user_input[loan_col]=1

# Feature scalling

user_scaled=scaler.transform(user_input)
prediction=model.predict(user_scaled)[0]

if prediction==1:
    print("\nloan_status: yes")
else:
    print("\nloan_status: no")
