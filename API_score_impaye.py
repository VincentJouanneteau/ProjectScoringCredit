import pandas as pd
import numpy as np
from flask import Flask,render_template, request, jsonify
import pickle
#from gevent.pywsgi import WSGIServer

app = Flask(__name__)
# Load the model
model = pickle.load(open('./mlflow_model/model.pkl','rb'))
xtest = pd.read_csv('xtest_norm.csv')
df = pd.read_csv('submission_file_name.csv')

app= Flask(__name__, template_folder='templates')

#app.config.from_object('config')
@app.route('/')
def home():
    return render_template('index.html')

   
@app.route('/score/<int:sk_id>')
def score(sk_id):

    all_id_client = list(df['SK_ID_CURR'].unique())
    ID = sk_id
    ID = int(ID)
    seuil = 0.084
    if ID not in all_id_client:
        prediction="Client inconnu"
        proba_impaye=0
    else :
        idx = int(df[df['SK_ID_CURR']==ID].index[0])
        X = xtest.loc[[idx]]

        #data = df[df.index == comment]
        proba_impaye = float(model.predict_proba(X)[:, 1])
        if proba_impaye >= seuil:
            prediction = "Refusé"
        else:
            prediction = "Accepté"
    
    #return render_template('index.html', prediction_text=prediction,prediction_score=proba_impaye)
    return jsonify({ 'prediction_text' : prediction, 'prediction_score': proba_impaye})
            

#lancement de l'application
if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8080,debug=True) 