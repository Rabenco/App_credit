import streamlit as st
import streamlit.components.v1 as components
import os
import datetime
from joblib import load
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly
import shap


# Title 
st.markdown("<h1 style='text-align: center; color: #E2383F;'><strong>📈 PRET A DEPENSER DASHBOARD</u></strong></h1>", unsafe_allow_html=True)
# Subtitle
st.markdown("<h4 style='text-align: center'><i>“A timely return of the loan makes it easier to borrow a second time.”</i></h4>", unsafe_allow_html=True)
st.markdown("***")

DATA_FILE = 'data_final.csv'

# Load
@st.cache(persist = True)
def load_all():
    
    def load_data():
        data = pd.read_csv(DATA_FILE)
        
        data_shap = data.drop(columns=['Prediction', 'Prediction_Score', 
        'PREDICTION_NEIGHBORS_20_MEAN','GENDER', 'NAME_CONTRACT_TYPE', 
        'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
        'NAME_HOUSING_TYPE','YEARS_BIRTH','YEARS_EMPLOYED'])
        #data_shap = data_shap.set_index('SK_ID_CURR')
        return data, data_shap
    # load data
    data, data_shap = load_data()
    
    def load_model():
        # load
        model = load('best_model.joblib')
        return model
    # Load model
    model = load_model()

    return data, data_shap, model

data, data_shap, model = load_all()

resultats = data.copy()

###############      APP        #######################
#
#-------------------------------------------------------
# Filter
#-------------------------------------------------------

# Espace : 2 colonnes, l'une deux fois plus grande que l'autre
left_column_0, _ = st.beta_columns((1, 3))

# Filtre clients
left_column_0.header("**Client ID**")
client_id = left_column_0.selectbox("Please select a client ID", data["SK_ID_CURR"])

#-------------------------------------------------------
# Information
#-------------------------------------------------------

# Infos principales client
client_info = data[data["SK_ID_CURR"]==client_id].iloc[0]
st.write(client_info.to_frame().T)

#-------------------------------------------------------
# Prediction Score
#-------------------------------------------------------
# Title
st.header("**Default Risk Score**")

# Obtenir le score de prediction du client
client_score = data[data["SK_ID_CURR"]==client_id].iloc[0]["Prediction_Score"]

# Seuil de solvabilité
st.write('*Select the threshold* : (default : **0.41**)')

def threshold():
    new_threshold = st.slider(
        label='Threshold:',
        min_value=0.,
        value=0.41,
        max_value=1.)
    preda_proba = data['Prediction_Score']/100
    # new predictions
    pred = (preda_proba >= new_threshold).astype('int')
    # update results
    resultats['Prediction'] = pred
    return new_threshold

current_threshold = threshold()

resultats["Prediction_name"]= resultats["Prediction"].replace({0 : 'Non defaulter', 1 : 'Defaulter'})
client_cible = resultats[resultats["SK_ID_CURR"]==client_id].iloc[0]["Prediction_name"]
st.info("**Prediction of the selected client with the current threshold : ** : **{}** ".format(client_cible))

# Obtenir le score de prédiction (moyenne) des 20 plus proches profiles clients 
similar_clients_credit_score = data[data["SK_ID_CURR"]==client_id].iloc[0]["PREDICTION_NEIGHBORS_20_MEAN"]

#https://plotly.com/python/gauge-charts/
gauge = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Prediction Score", 'font': {'size': 24}},
        value = client_score,
        mode = "gauge+delta+number",
        
        gauge = {'axis': {'range': [None, 100]},
                 'steps' : [
                     {'range': [0, 20], 'color': "#008177"},
                     {'range': [20, 40], 'color': "#00BAB3"},
                     {'range': [40, 50], 'color': "#D4E88B"},
                     {'range': [50, 60], 'color': "#F4EA9D"},
                     {'range': [60, 80], 'color': "#FF9966"},
                     {'range': [80, 100], 'color': "#E2383F"},
                     ],
                 'threshold': {
                'line': {'color': "black", 'width': 10},
                'thickness': 0.8,
                'value': client_score},

                 'bar': {'color': "black", 'thickness' : 0.15},
                },
        delta = {'reference': similar_clients_credit_score,
        'increasing': {'color': '#E2383F'},
        'decreasing' : {'color' : '#008177'}}
        ))

gauge.update_layout(width=600, height=500, 
                    margin=dict(l=50, r=50, b=100, t=100, pad=4))

st.plotly_chart(gauge)

if 0 < client_score <= 20:
    trust_text = "EXCELLENT"
elif 20 < client_score <= 40:
    trust_text = "GOOD"
elif 40 < client_score <= 60:
    trust_text = "AVERAGE"
elif 60 < client_score <= 80:
    trust_text = "LOW"
else :
    trust_text = "WEAK"

left_column_1, _ = st.beta_columns((3, 2))
left_column_2, _ = st.beta_columns((3, 1))
left_column_1.info('TRUST score for the selected client : **{}**'.format(trust_text))
left_column_1.markdown('Prediction Score for similar clients : **{0:.1f}**'.format(similar_clients_credit_score))


###############      SIDEBAR        #######################

# Chargement du logo
logo =  Image.open("Images/pret_logo.png") 
st.sidebar.image(logo,
                width=240,
                caption="Now you can")

# Préparation informations client sélectionné

client_age = client_info["YEARS_BIRTH"]
client_employed = client_info["YEARS_EMPLOYED"]
client_work = client_info["NAME_INCOME_TYPE"]
client_income = client_info["AMT_INCOME_TOTAL"]
client_contract = client_info["NAME_CONTRACT_TYPE"]
client_status = client_info["NAME_FAMILY_STATUS"]
client_gender = client_info["GENDER"]
client_education = client_info["NAME_EDUCATION_TYPE"]


# Affichage d'informations sur le client sélectionné dans la sidebar
st.sidebar.header("📋 Client informations")
st.sidebar.write("**Age**", int(client_age), "years")
st.sidebar.write("**Gender** :", client_gender)
st.sidebar.write("**Family status** :", client_status)
st.sidebar.write("**Education** :", client_education)
st.sidebar.write("**Years employed**", int(client_employed), "years")
st.sidebar.write("**Income type** :", client_work)
st.sidebar.write("**Income**", int(client_income), "$")
st.sidebar.write("**Contract type** :", client_contract) 


# Préparation informations clientèle totale
client_age_mean = data['YEARS_BIRTH'].mean()
client_age_min = data['YEARS_BIRTH'].min()
client_age_max = data['YEARS_BIRTH'].max()
client_employed_mean = data['YEARS_EMPLOYED'].mean()
client_employed_min = data['YEARS_EMPLOYED'].min()
client_employed_max = data['YEARS_EMPLOYED'].max()
client_income_mean = data['AMT_INCOME_TOTAL'].mean()
client_income_min = data['AMT_INCOME_TOTAL'].min()
client_income_max = data['AMT_INCOME_TOTAL'].max()


# Sélection d'informations générales sur le client
st.sidebar.header("📊 More informations")
st.sidebar.subheader("Stats and client infos")

# Age
if st.sidebar.button("Age"):
    st.sidebar.write("**Age** :", int(client_age), "years")
    # Graph dans app principale
    if st.sidebar.checkbox("Show age infos / Hide", value = True):
        left_column_1.header("**Client : age**")
        left_column_1.success("**Age of the selected client** : **{}** years".format(int(client_age)))
        fig = go.Figure(go.Indicator(
            mode="number+gauge+delta",
            value=client_age,
            delta={
                'reference': client_age_mean,
                'increasing': {'color': '#77C5D5'},
                'decreasing': {'color': '#0093B2'}
            },
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text':
                "<b>Age</b><br><span style='color: gray; font-size:0.8em'>mean : 44</span>",
                'font': {"size": 16}
            },
            gauge={'shape': "bullet",
                'axis': {'range': [client_age_min, client_age_max]},
                'threshold': {'line': {'color': "red",'width': 2},
                'thickness': 0.75,
                'value': client_age
            },
                'steps': [{'range': [client_age_min, client_age_mean], 'color': "#0093B2"}, 
                {'range': [client_age_mean, client_age_max], 'color': "#B8DDE1"}],
                'bar': {'color': "#002E5D"}
            }))
        fig.update_layout(width=700, height=300, margin=dict(l=100, r=50, b=100, t=100, pad=4))
        st.plotly_chart(fig)

# Code gender
if st.sidebar.button("Gender"):
    st.sidebar.write("**Gender** :", client_gender)
    if st.sidebar.checkbox("Show gender / Hide", value = True):
        # Graph dans app principale
        left_column_1.header("**Stats : gender**")
        left_column_1.success("**Gender of the selected client** : **{}** ".format(client_gender))
        st.subheader('Distribution of code gender for all targets')
        img_status =  Image.open("Images/code_gender_1.png") 
        st.image(img_status,
                    width=400)
        st.subheader('Percentage of **defaulters** for each category of code gender')            
        img_status =  Image.open("Images/code_gender_2.png") 
        st.image(img_status,
                    width=400)  

# Family status
if st.sidebar.button("Family status"):
    st.sidebar.write("**Family status** :", client_status)
    if st.sidebar.checkbox("Show Family status infos / Hide", value = True):
        # Graph dans app principale
        left_column_1.header("**Stats : family status**")
        left_column_1.success("**Family status of the selected client** : **{}** ".format(client_status))
        st.subheader('Distribution of family status')
        img_status =  Image.open("Images/status_1.png") 
        st.image(img_status,
                    width=600)
        st.subheader('Percentage of **defaulters** for each category of family status')            
        img_status =  Image.open("Images/status_2.png") 
        st.image(img_status,
                    width=600)           
# Education
if st.sidebar.button("Education "):
    st.sidebar.write("**Education** :", client_education)
    if st.sidebar.checkbox("Show education infos / Hide", value = True):
        # Graph dans app principale
        left_column_1.header("**Stats : Education type**")
        left_column_2.success("**Education type of the selected client** : **{}** ".format(client_education))
        st.subheader('Distribution of education type')
        img_status =  Image.open("Images/education_1.png") 
        st.image(img_status,
                    width=600)
        st.subheader('Percentage of **defaulters** for each category of education')            
        img_status =  Image.open("Images/education_2.png") 
        st.image(img_status,
                    width=600)      

# Emploi : durée
if st.sidebar.button("Years employed"):
    st.sidebar.write("**Years employed**", int(client_employed), "years")
    if st.sidebar.checkbox("Show years employed infos / Hide", value = True):
        # Graph dans app principale
        left_column_1.header("**Client : years employed**")
        left_column_1.success("**Years employed of the selected client** : **{}** ".format(int(client_employed)))
        fig = go.Figure(go.Indicator(
            mode="number+gauge+delta",
            value=client_employed,
            delta={
                'reference': client_employed_mean,
                'increasing': {'color': '#77C5D5'},
                'decreasing': {'color': '#0093B2'}
            },
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text':
                "<b>Years employed</b><br><span style='color: gray; font-size:0.8em'>mean : 6.4</span>",
                'font': {"size": 16}
            },
            gauge={'shape': "bullet",
                'axis': {'range': [client_employed_min, client_employed_max]},
                'threshold': {'line': {'color': "red",'width': 2},
                'thickness': 0.75,
                'value': client_employed
            },
            'steps': [{'range': [client_employed_min, client_employed_mean], 'color': "#0093B2"}, 
                {'range': [client_employed_mean, client_employed_max], 'color': "#B8DDE1"}],
                'bar': {'color': "#002E5D"}
            }))
        fig.update_layout(width=700, height=300, margin=dict(l=180, r=50, b=100, t=100, pad=4))
        st.plotly_chart(fig)    

# Revenus du client
if st.sidebar.button("Income type"):
    st.sidebar.write("**Income type** :", client_work)
    # Graph dans app principale
    if st.sidebar.checkbox("Show income type infos / Hide", value = True):
        left_column_1.header("**Stats : income type**")
        left_column_1.success("**Income type of the selected client** : **{}** ".format(client_work))
        st.subheader('Distribution of income type')
        img_status =  Image.open("Images/income_type_1.png") 
        st.image(img_status,
                    width=400)
        st.subheader('Percentage of **defaulters** for each category of income type')
        img_status =  Image.open("Images/income_type_2.png") 
        st.image(img_status,
                    width=400)

if st.sidebar.button("Income"):
    st.sidebar.write("**Income**", int(client_income), "$")
     # Graph dans app principale
    if st.sidebar.checkbox("Show income infos / Hide", value = True): 
        left_column_1.header("**Client : income**")
        left_column_1.success("**Income of the selected client** : **{}** ".format(int(client_income)))
        fig = go.Figure(go.Indicator(
            mode="number+gauge+delta",
            value=client_income,
            delta={
                'reference': client_income_mean,
                'increasing': {'color': '#77C5D5'},
                'decreasing': {'color': '#0093B2'}
            },
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text':
                "<b>Income</b><br><span style='color: gray; font-size:0.8em'>mean : 0.17M</span>",
                'font': {"size": 16}
            },
            gauge={'shape': "bullet",
                'axis': {'range': [client_income_min, client_income_max]},
                'threshold': {'line': {'color': "red",'width': 2},
                'thickness': 0.75,
                'value': client_income
            },
                'steps': [{'range': [client_income_min, client_income_mean], 'color': "#0093B2"}, 
                {'range': [client_income_mean, client_income_max], 'color': "#B8DDE1"}],
                'bar': {'color': "#002E5D"}
            }))
        fig.update_layout(width=700, height=300, margin=dict(l=110, r=50, b=100, t=100, pad=4))
        st.plotly_chart(fig) 

if st.sidebar.button("Contract type"):
    st.sidebar.write("**Contract type** :", client_contract)      
    # Graph dans app principale
    if st.sidebar.checkbox("Show type of loans stats / Hide", value = True):
        left_column_1.header("**Stats : contract type**")
        left_column_1.success("**Type of loan of the selected client** : **{}** ".format(client_contract))
        st.subheader('Distribution of contract type')
        img_status =  Image.open("Images/contract_1.png") 
        st.image(img_status,
                    width=400)
        st.subheader('Percentage of **defaulters** of contract type for all targets')
        img_status =  Image.open("Images/contract_2.png") 
        st.image(img_status,
                    width=400)

###############         SHAP         #######################

st.sidebar.subheader('📈 SHAP explainer')
def shap_explainer():
    if st.sidebar.button("Explain Results by SHAP"):
        with st.spinner('** ⏳ Calculating shap values...**'):
            st.header("**SHAP : explain results**")
            st.markdown("<h4 style='text-align: center'><strong>How most important features impacts Class prediction?</strong></h4>", unsafe_allow_html=True)
            st.write('*__Force plot__ shows, how opposite are the features strenghs*')
            if st.sidebar.checkbox("SHAP / Hide", value = True):
                st.write('**SHAP HELP**')
                img_help_shap =  Image.open("Images/shap_explain.png") 
                st.image(img_help_shap,
                        width=700)
    
            explainer = shap.TreeExplainer(model)
            row_to_show = data_shap[data_shap["SK_ID_CURR"]==client_id].index
            data_cal = data_shap.set_index("SK_ID_CURR")
            data_for_prediction = data_cal.iloc[row_to_show]  # use 1 row of data here. Could use multiple rows if desired
            data_for_prediction_array = data_for_prediction.values.reshape(1, -1)
            # Calculate Shap values
            shap_values = explainer.shap_values(data_for_prediction_array)
            ind_fig = shap.force_plot(explainer.expected_value[1], 
                            shap_values[1], 
                            data_for_prediction, 
                            plot_cmap=["#EF553B","#636EFA"])
            ind_fig_html = f"<head>{shap.getjs()}</head><body>{ind_fig.html()}</body>"
            st.write('**SHAP Force plot for the selected client**')
            components.html(ind_fig_html, height=120)
shap_explainer()
