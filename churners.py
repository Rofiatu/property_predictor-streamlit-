import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score as acc
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('BankChurners.csv')
data.drop(['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
       'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2', 'CLIENTNUM', 'Income_Category'], axis = 1, inplace = True)

cat = data.select_dtypes(include = ['category', 'object'])
num = data.select_dtypes(include = 'number' )

x = data.drop(['Attrition_Flag'], axis = 1)
y = data.Attrition_Flag

print(f"Categorical Dataset \n {cat.head()}\n \n ")
print(f"Numerical Dataset \n {num.head()}\n \n \n")
print(f"General Dataset \n {data.head()}\n \n")

encoder = LabelEncoder()
for i in cat.columns:
    if i in data.columns:
        data[i] = encoder.fit_transform(data[i])

scaler = StandardScaler()
for i in num.columns:
    if i in data.columns:
        data[i] = encoder.fit_transform(data[i])


x_trans = data.drop(['Attrition_Flag'], axis = 1)
y_trans = data.Attrition_Flag

best_feature1 = SelectKBest(score_func = f_classif, k = 'all')
fitting1 = best_feature1.fit(x_trans,y_trans)
scores1 = pd.DataFrame(fitting1.scores_)
columns1 = pd.DataFrame(x.columns)
feat_score1 = pd.concat([columns1, scores1], axis = 1)
feat_score1.columns = ['Feature', 'F_classif_score'] 
selected = feat_score1.nlargest(10, 'F_classif_score')
selected.sort_values(by = 'F_classif_score', ascending = False)


sel_feat = data[['Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1',       'Total_Trans_Amt',
       'Total_Revolving_Bal', 'Contacts_Count_12_mon',
       'Avg_Utilization_Ratio', 'Months_Inactive_12_mon']]

# Split data into train and test
# Notice that i pass in 'y', and not y_trans.
# coz y is still in words, so my prediction will be straight forward
x_train , x_test , y_train, y_test = train_test_split(sel_feat, y, test_size = 0.2, stratify = y)

randForest = RandomForestClassifier()
randForest.fit(x_train, y_train)
validation = randForest.predict(x_train)
prediction = randForest.predict(x_test)

acc(y_test, prediction)

# Save Model
import joblib
joblib.dump(randForest, 'bankChurn_model.pkl')


# ------------Streamlit Production Starts -----------------

st.markdown("<h1 style = 'text-align: right; color: #B2A4FF'>CHURNERS TEST INTERFACE</h1> ", unsafe_allow_html = True)
st.markdown("<h6 style = 'top_margin: 0rem; text-align: right; color: #FFB4B4'>Built by GoMyCode Pumpkin Reedemers</h6>", unsafe_allow_html = True)

img1 = st.image('images\pngwing.com (21).png')

st.write('Pls register your name for record of usage')
username = st.text_input('Enter your name')
if st.button('submit name'):
    st.success(f"Welcome {username}. Pls use according to usage guidelines")

side_img1 = st.sidebar.image('images\pngwing.com (26).png', caption = username, width = 200)

entry = st.sidebar.selectbox('How do you want to enter your variables', ['Direct Input', 'Slider']) 

if entry == 'Direct Input':
    Total_Trans_Ct = st.sidebar.number_input('Total_Trans_Ct',min_value = 10, max_value = 140)
    Total_Ct_Chng_Q4_Q1 = st.sidebar.number_input('Total_Ct_Chng_Q4_Q1',min_value = 0, max_value = 4)
    Total_Trans_Amt = st.sidebar.number_input('Total_Trans_Amt',min_value = 510, max_value = 18500)
    Total_Revolving_Bal = st.sidebar.number_input('Total_Revolving_Bal',min_value = 0, max_value = 2520)
    Contacts_Count_12_mon = st.sidebar.number_input('Contacts_Count_12_mon',min_value = 0, max_value = 7)
    Avg_Utilization_Ratio = st.sidebar.number_input('Avg_Utilization_Ratio',min_value = 0, max_value = 1)
    Months_Inactive_12_mon = st.sidebar.number_input('Months_Inactive_12_mon',min_value = 0, max_value = 8)

elif entry == 'Slider':
    Total_Trans_Ct = st.sidebar.slider('Total_Trans_Ct',10, 140, value = 100, on_change=None)
    Total_Ct_Chng_Q4_Q1 = st.sidebar.slider('Total_Ct_Chng_Q4_Q1', 0,  4, value = 2, on_change=None)
    Total_Trans_Amt = st.sidebar.slider('Total_Trans_Amt', 510, 18500, value = 10000, on_change=None)
    Total_Revolving_Bal = st.sidebar.slider('Total_Revolving_Bal', 0,  2520, value = 1000, on_change=None)
    Contacts_Count_12_mon = st.sidebar.slider('Contacts_Count_12_mon', 0,  7, value = 4, on_change=None)
    Avg_Utilization_Ratio = st.sidebar.slider('Avg_Utilization_Ratio', 0.0, 1.0, value = 0.5, on_change=None, step = 0.01)
    Months_Inactive_12_mon = st.sidebar.slider('Months_Inactive_12_mon', 0,  7, value = 5, on_change=None)

input_variables = [[Total_Trans_Ct, Total_Ct_Chng_Q4_Q1, Total_Trans_Amt, Total_Revolving_Bal, Contacts_Count_12_mon,
    Avg_Utilization_Ratio, Months_Inactive_12_mon]]

frame = ({'Total_Trans_Ct':[Total_Trans_Ct], 
          'Total_Trans_Ct':[Total_Trans_Ct], 'Total_Ct_Chng_Q4_Q1':[Total_Ct_Chng_Q4_Q1], 'Total_Trans_Amt':[Total_Trans_Amt], 'Total_Revolving_Bal':[Total_Revolving_Bal], 'Contacts_Count_12_mon':[Contacts_Count_12_mon],
        'Avg_Utilization_Ratio':[Avg_Utilization_Ratio], 'Months_Inactive_12_mon':[Months_Inactive_12_mon]
        })

st.markdown("<hr><hr>", unsafe_allow_html= True)

st.write('These are your input variables')
frame = pd.DataFrame(frame)
frame = frame.rename(index = {0: 'Value'})
frame = frame.transpose()
st.write(frame)

# load the model
model = joblib.load(open('bankChurn_model.pkl', 'rb'))
model_pred = model.predict(input_variables)
proba_scores = model.predict_proba(input_variables)

from datetime import date
today_date = date.today()

if st.button('PREDICT'):
    if model_pred == 'Existing Customer':
        st.success('Existing Customer')
        st.text(f"Probability Score: {proba_scores}")
        st.image('images\pngwing.com (23).png', caption = 'EXISTING', width = 200)
        st.write('Here are some additional ideas to further keep existing customers: ', ['Offer loyalty programs', 'Provide exceptional customer service:', 'Ask for feedback', 'Create special promotions that are only available to existing customers', 'Communicate regularly', 'Provide excellent after-sales support'])
        st.info(f"predicted at: {today_date}")
        
    else:
        st.warning('Attrited Customer')
        st.text(f"Probability Score: {proba_scores}")
        st.image('images\pngwing.com (27).png', caption = 'ATTRITED', width = 200)
        st.write('Here are some ideas to keep the customer', ['Personalize the customer experience', 'Listening to your customers complaints and concerns and address them in a timely manner.', 'Improve the product or service', 'Offer incentives to customers for staying loyal to your business. This could include rewards programs, discounts, and other exclusive offers', 'Provide excellent customer service', 'Keep in touch'])
        st.info(f"predicted at: {today_date}")


st.markdown("<br>", unsafe_allow_html = True )
# - Check for Multi-Colinearity in the chosen features.
heat = plt.figure(figsize = (14, 7))
sns.heatmap(sel_feat.corr(), annot = True, cmap = 'BuPu')

st.write(heat)



