import pandas as pd
import numpy as np
from flask import Flask,render_template, request, jsonify
import pickle


app = Flask(__name__)
# Chargement du modèle
model = pickle.load(open('./mlflow_model/model.pkl','rb'))
xtest = pd.read_csv('xtest_norm.csv')
df = pd.read_csv('submission_file_name.csv')

app= Flask(__name__, template_folder='templates')

#app.config.from_object('config')
@app.route('/')
def home():
    return render_template('index.html')

#Fonction renvoyant la prediction et le score impaye   
@app.route('/score/<int:sk_id>')
def score(sk_id):

    all_id_client = list(df['SK_ID_CURR'].unique())
    ID = sk_id
    ID = int(ID)
    #Seuil de modèle XGBOOST
    seuil = 0.084
    #Test si client reconnu dans la liste
    if ID not in all_id_client:
        prediction="Client inconnu"
        proba_impaye=0
    else :
        idx = int(df[df['SK_ID_CURR']==ID].index[0])
        X = xtest.loc[[idx]]
        #Calcul du score impaye : proba appartenir à la classe 1
        proba_impaye = float(model.predict_proba(X)[:, 1])
        #Prediction en fonction du score et du seuil
        if proba_impaye >= seuil:
            prediction = "Refusé"
        else:
            prediction = "Accepté"
    
    #renvoi la prediction et la proba impayé
    return jsonify({ 'prediction_text' : prediction, 'prediction_score': proba_impaye})
            

#lancement de l'application sur le port 8080 #ouverte sur le serveur AWS
if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8080,debug=True) 