from flask import Flask,request,jsonify,render_template,url_for
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

application=Flask(__name__)
app=application
model=pickle.load(open('Model_Diabeter.pkl','rb'))
@app.route("/")
def index():
    return render_template("index.html")
@app.route('/predictdiabetes',methods=['GET','POST'])
def predict_datapoint():
    if request.method=="POST":
        
        Pregnancies = float(request.form.get('Pregnancies'))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = float(request.form.get('Age'))

        new_data_scaled=[[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]]
        result=model.predict(new_data_scaled)
        if result==0:
            result="Not a Diabetic Patient"
        else:
            result="Diabetic Patient"
        print(result)

        return render_template('home.html',results=result)


    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")