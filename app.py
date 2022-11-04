from flask import Flask, url_for, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('static/models/logistic_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/classifier')
def classifier():
    return render_template('classifier.html')

@app.route('/result',methods=['POST','GET'])
def result():
    if request.method == 'POST':
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        chestpaintype = int(request.form['chestpaintype'])
        restingbp = int(request.form['restingbp'])
        cholesterol = int(request.form['cholesterol'])
        fastingbs = int(request.form['fastingbs'])
        restingecg = int(request.form['restingecg'])
        maxhr = int(request.form['maxhr'])
        angina = int(request.form['angina'])
        oldpeak = int(request.form['oldpeak'])
        stslope = int(request.form['stslope'])

        data = np.array([[age, sex, chestpaintype, restingbp, cholesterol, fastingbs, restingecg, maxhr, angina, oldpeak, stslope]])
        prediction = model.predict(data)
        if prediction[0] == 0:
            my_prediction = "No possibility of Heart Failure"
        elif prediction[0] == 1:
            my_prediction = "Possibility of Heart Failure"
        return render_template('result.html',prediction = my_prediction)
    return "<html>404 Error Page</html>"



if __name__== '__main__':
    app.run(debug=True)