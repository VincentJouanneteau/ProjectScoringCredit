import plotly.graph_objects as go
import pandas as pd
import shap
import requests
import plotly.graph_objects as go
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pickle
#import mlflow
#from streamlit_shap import st_shap
import lime
import lime.lime_tabular
import streamlit.components.v1 as components
import plotly.express as px


st.set_option('deprecation.showPyplotGlobalUse', False)
#st.set_page_config(layout="wide")

#Chargement des données Train / Test
#Mise en cache chargement data
@st.cache_data
def chargement_files():
    df = pd.read_csv('submission_file_name.csv')
    xtest = pd.read_csv('xtest_norm.csv')
    xtest_brut = pd.read_csv('xtest.csv')
    xtrain = pd.read_csv('xtrain_.csv')
    ytrain = pd.read_csv('target.csv',nrows=10000)
    
    return df,xtest,xtest_brut,xtrain,ytrain

df,xtest,xtest_brut,xtrain,ytrain = chargement_files()

#Chargement / Calcul du model pkl -- Mis dans une fonction à part pour mise en cache
#Mise en cache ressource
@st.cache_resource
def chargement_model() :
    path_model = './mlflow_model/model.pkl'
    loaded_model = pickle.load(open(path_model, 'rb'))
    ytest_pred = loaded_model.predict_proba(xtest)[:,1]
    explainer1 = shap.Explainer(loaded_model.predict,xtest[:10000])
    return loaded_model,ytest_pred,explainer1

loaded_model,ytest_pred,explainer1 = chargement_model()    

#Calcul Explainer pour lime et shap -- Mis dans une fonction à part pour mise en cache
#Mise en cache ressource
@st.cache_resource
def calcul_explainer() :
    # explainer feature importance local lime
    feats = xtest.columns
    explainer = lime.lime_tabular.LimeTabularExplainer(xtrain[feats].values,mode='classification',training_labels=ytrain['TARGET'],
                                                       feature_names=xtest.columns.values.tolist())
    shap_values = explainer1(xtest[:10000]) 
    # explainer feature importance global shap
    return feats,explainer,shap_values

feats,explainer,shap_values = calcul_explainer()
    
#Calcul des scores pour le jeu de test
xtest_brut['TARGET_proba'] = ytest_pred

def calcul_octroi(predict_proba):
    seuil = 0.084
    if predict_proba>seuil:
        decision = 1
    else:
        decision = 0
    return decision

#Calcul de la TARGET 0 / 1 selon le seuil et la proba 
xtest_brut['TARGET'] = xtest_brut['TARGET_proba'].apply(calcul_octroi)



st.write("""
# JAUGE ET FEATURE IMPORTANCE LOCAL avec LIME
""")

# Jauge d'acceptation/Refus
def maj_jauge(sk_id):

    if sk_id in list_client_id:
        # Calcul de la prédiction impayé et acceptation/refus à partir de l'API Flask stocké dans AWS / EC2
        url_api = "http://13.38.119.19:8080/score/" + str(sk_id)
        response = requests.get(url_api)
        if response:
            decision = response.json()['prediction_text']
            predict_proba = float(response.json()['prediction_score'])
        else:
            print("erreur accès API : ", response)
        
        jauge_predict = go.Figure(go.Indicator( mode = "gauge+number",
                                            value = predict_proba,
                                            domain = {'x': [0, 1], 'y': [0, 1]},                                                                                   
                                            gauge = {
                                                'axis': {'range': [0, 0.3], 'tickwidth': 1, 'tickcolor': "black"},
                                                'bar': {'color': "orange"},
                                                'bgcolor': "white",
                                                'borderwidth': 2,
                                                'bordercolor': "gray",   
                                                'steps': [
                                                    {'range': [0, 0.086], 'color': 'lightgreen'},
                                                    {'range': [0.087, 0.3], 'color': 'lightcoral'}],
                                                'threshold': {
                                                    'line': {'color': "red", 'width': 4},
                                                    'thickness': 1,
                                                    'value': 0.086}},
                                            title = {'text': f"client {sk_id} décision : {decision}"}))

        return jauge_predict

list_client_id = df['SK_ID_CURR'].tolist()
option_sk = st.selectbox('Choix numéro de client',list_client_id)

fig = maj_jauge(option_sk)
st.plotly_chart(fig)

    
# Graphique Lime pour la feature importance de l'individu sélectionné
#feats = xtest.columns
#explainer = lime.lime_tabular.LimeTabularExplainer(xtrain[feats].values,mode='classification',training_labels=ytrain['TARGET'],/
#                                                   feature_names=xtest.columns.values.tolist())

idx = int(df[df['SK_ID_CURR']==option_sk].index[0])

exp = explainer.explain_instance(xtest.values[idx],loaded_model.predict_proba,num_features=8)
html_lime = exp.as_html()
components.html(html_lime, width=1100, height=350, scrolling=True)

#Récupération des variables sélectionnées pour le modèle
list_variable_norm = xtest.columns.values.tolist()
list_variable = [ele[:-2] for ele in list_variable_norm]

st.write("""
# Analyse par Variable
""")

#Choix variable 1 
option_variable1 = st.selectbox('Choix variable 1',list_variable)

# Boite à moustache de la variable 1
X = 'TARGET'
Y = option_variable1

fig = plt.figure(figsize=(8,5))
#Data nécessaire pour le box plot en excluant les valeurs vides
sous_echantillon = xtest_brut[[X,Y]]
sous_echantillon = sous_echantillon[sous_echantillon[Y].notnull()]

modalites = sous_echantillon[X].drop_duplicates().sort_values(ascending=False)
libelle_modalites = ['Refusé','Accepté']
groupes = []
for m in modalites:
    groupes.append(sous_echantillon[sous_echantillon[X]==m][Y])

# Propriétés graphiques (pas très importantes)    
medianprops = {'color':"blue"}
meanprops = {'marker':'o', 'markeredgecolor':'black',
            'markerfacecolor':'firebrick'}

plt.ylabel('Catégorie',fontsize=12)
plt.xlabel(option_variable1,fontsize=12)
plt.title(option_variable1 + ' selon acceptation/refus',fontsize=14)
plt.boxplot(groupes, labels=libelle_modalites, showfliers=False, medianprops=medianprops, 
            vert=False, patch_artist=True, showmeans=True, meanprops=meanprops)

result_var1 = xtest_brut.loc[xtest_brut.index[idx],Y]
plt.axvline(x = result_var1, color = "LightSeaGreen", label = 'axvline - full height')

st.pyplot(fig)

#Choix variable 1 
option_variable2 = st.selectbox('Choix variable 2',list_variable)

# Boite à moustache de la variable 2
X = 'TARGET'
Y = option_variable2

fig1 = plt.figure(figsize=(8,5))
#Data nécessaire pour le box plot en excluant les valeurs vides
sous_echantillon = xtest_brut[[X,Y]]
sous_echantillon = sous_echantillon[sous_echantillon[Y].notnull()]
modalites = sous_echantillon[X].drop_duplicates().sort_values(ascending=False)
libelle_modalites = ['Refusé','Accepté']
groupes = []
for m in modalites:
    groupes.append(sous_echantillon[sous_echantillon[X]==m][Y])

medianprops = {'color':"blue"}
meanprops = {'marker':'o', 'markeredgecolor':'black',
            'markerfacecolor':'firebrick'}

plt.ylabel('Catégorie',fontsize=12)
plt.xlabel(option_variable2,fontsize=12)
plt.title(option_variable2 + ' selon acceptation/refus',fontsize=14)
plt.boxplot(groupes, labels=libelle_modalites, showfliers=False, medianprops=medianprops, 
            vert=False, patch_artist=True, showmeans=True, meanprops=meanprops)

result_var2 = xtest_brut.loc[xtest_brut.index[idx],Y]
plt.axvline(x = result_var2, color = "LightSeaGreen", label = 'axvline - full height')

st.pyplot(fig1)


#Nuage de points croisant les 2 variables avec couleur dégradé selon la proba d'être refusé


fig3 = px.scatter(
    xtest_brut,
    x=option_variable1,
    y=option_variable2,
    opacity=0.15,
    color="TARGET_proba",
    color_continuous_scale=[(0, "green"), (0.1, "orange"), (0.2, "red"), (1, "black")],
)

fig3.update_traces(marker_size=3)

#Affichage de l'individu choisi
fig3.add_scatter(x=[result_var1], y=[result_var2],opacity=1, mode="markers",
                marker=dict(size=15, color="black"),name="client sélectionné")
 
    
fig3.add_annotation(x=result_var1, y=result_var2,
                   text='notre client',
                   showarrow=True,
                   arrowhead=1,
                   arrowsize=1,
                   arrowwidth=2,
                   arrowcolor='black',
                   bgcolor="LightSeaGreen",
                   borderwidth=1,
                   yshift=10,
                   font=dict(
                   color="black"
            ))


st.plotly_chart(fig3, theme="streamlit",use_container_width=True)

st.write("""
# Feature Importance Globale
""")

# Feature Importance Globale
#explainer = shap.Explainer(loaded_model.predict,xtest[:1000])
#shap_values = explainer(xtest[:1000]) 
fig2, ax = plt.subplots(figsize=(8, 12))
plt.title('Feature Importance Globale',fontsize=14)
ax.set_xlim(-2, 2)
fig2 = shap.plots.beeswarm(shap_values, max_display=10)
st.pyplot(fig2)
#st_shap(shap.plots.beeswarm(shap_values, max_display=10))    

