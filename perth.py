import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score as acc
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from datetime import date
import joblib
import io
import time
import warnings
warnings.filterwarnings('ignore')

# import data
perth = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vRDEbw34qTXRY-N-2-1meoFrp-2_7oihgjo-NZh-a2vIoOxK3LLnDvM5VvBU-iUZiz4S2YWMxQlFsRy/pub?output=csv')

# drop irrelevant columns
perth.drop(['ADDRESS','LATITUDE','LONGITUDE'],axis=1,inplace=True)

# check for null values in the existing columns
perth.isnull().sum().sort_values(ascending=False)

# categorize data based on content of columns
categorical = perth.select_dtypes(include='object')
numerical = perth.select_dtypes(exclude='object')

# ensure that features have been correctly classified as categorical or numerical
# change the date column to date type then rerun the categorization into categorical and numerical
perth['DATE_SOLD'] = pd.to_datetime(perth['DATE_SOLD'])
categorical = perth.select_dtypes(include='object')
numerical = perth.select_dtypes(exclude='object')

# fill in null values in numerical with mean (since there are no missing values in categorical)
for item in numerical:
    perth[item].fillna(perth[item].mean(),inplace=True)

# categorise houses into bins based on price ranges
# 1. first, identify the min and max of the price column
perth['PRICE'].describe() # minimum = 51,000 ; maximum = 2,440,000 ; average = 637,000

# 2. create bins for price taking the min, max and average into consideration
bins = [0,500000,700000,2550000] # this creates 3 groups, as follows: [0 - 500,000], [500,001 - 700,000], [700,001 - 2,550,000]

# 3. categorise the price column into bins
perth['PRICE_GROUP'] = pd.cut(perth['PRICE'], bins, labels=['low_cost', 'affordable', 'expensive'])
perth['PRICE_GROUP'].value_counts()

# recategorise your data into numerical and categorical
categorical = perth.select_dtypes(include = ['category','object'])
numerical = perth.select_dtypes(include = 'number')

# label encode categorical data
lb = LabelEncoder()
for item in categorical:
    perth[item] = lb.fit_transform(perth[item])

# standardize numerical data using MinMax scaler
scaler = MinMaxScaler()
for item in numerical:
    perth[[item]] = scaler.fit_transform(perth[[item]])

# split into x and y
x = perth.drop(['PRICE_GROUP','PRICE','DATE_SOLD'],axis=1)
y = perth['PRICE_GROUP']

# FEATURE SELECTION
# feature selection using ANOVA F-value
best_feature1 = SelectKBest(score_func = f_classif, k = 'all')
fitting1 = best_feature1.fit(x,y)
scores1 = pd.DataFrame(fitting1.scores_)
columns1 = pd.DataFrame(x.columns)
feat_score1 = pd.concat([columns1, scores1], axis = 1)
feat_score1.columns = ['Feature', 'F_classif_score'] 
k1 = feat_score1.nlargest(14, 'F_classif_score')

k1.sort_values(by = 'F_classif_score', ascending = False)

# initialise selected features as a variable
sel_features = perth[['FLOOR_AREA','NEAREST_SCH_RANK','BATHROOMS','CBD_DIST','BEDROOMS','BUILD_YEAR','GARAGE','POSTCODE','NEAREST_STN_DIST','LAND_AREA']]
# sel_features

# MODELLING
# split new dataset into train and test
x_train, x_test, y_train, y_test = train_test_split(sel_features, y, test_size = 0.2, random_state = 20, stratify=y)

# modelling using xgboost
xgb_model = xgb.XGBClassifier()
xgb_model.fit(x_train, y_train)

# cross validate
xgb_val = xgb_model.predict(x_train)
print(classification_report(y_train, xgb_val))

#predict on the test dataframe to ascertain the accuracy of the model
xgb_prediction = xgb_model.predict(x_test)
print(classification_report(y_test, xgb_prediction))

revealer = confusion_matrix(xgb_prediction, y_test)
sns.set(style = 'darkgrid')
sns.heatmap(revealer/np.sum(revealer), annot=True, cmap='crest', fmt='.1%', linewidth=1)

#print results and comment on fitting accuracy
print(f'Accuracy of train data is: {acc(y_train,xgb_val)}')
print(f'Accuracy of test data is: {acc(y_test,xgb_prediction)}')
print(f'Thus, there is no over-fitting in the train and test dataset')

# modelling using random_forest
rfm = RandomForestClassifier()
rfm.fit(x_train, y_train)

rfm_val = rfm.predict(x_train)
print(classification_report(y_train, rfm_val))

# predict on the test dataframe to ascertain the accuracy of the model
rf_prediction = rfm.predict(x_test) #....................... Check the model performance on on new (test) dataset
print(classification_report(y_test, rf_prediction))

revealer = confusion_matrix(rf_prediction, y_test)
sns.set(style = 'darkgrid')
sns.heatmap(revealer/np.sum(revealer), annot=True, cmap='crest', fmt='.1%', linewidth=1)

#print results and comment on fitting accuracy
print(f'Accuracy of train data is: {acc(y_train,rfm_val)}')
print(f'Accuracy of test data is: {acc(y_test,rf_prediction)}')
print(f'Thus, there is over-fitting in the train and test dataset')
print(f'Conclusion: We will rely on the xgboost model for this task')

# modelling using logistic regression
logistic = LogisticRegression()
logistic.fit(x_train,y_train)

log_val = logistic.predict(x_train)
print(classification_report(y_train, log_val))

# predict on test dataframe to ascertain the accuracy of the model
log_prediction = logistic.predict(x_test)
print(classification_report(y_test, log_prediction))

revealer = confusion_matrix(log_prediction, y_test)
sns.set(style = 'darkgrid')
sns.heatmap(revealer/np.sum(revealer), annot=True, cmap='crest', fmt='.1%', linewidth=1)

#print results and comment on fitting accuracy
print(f'Accuracy of train data is: {acc(y_train,log_val)}')
print(f'Accuracy of test data is: {acc(y_test,log_prediction)}')
print(f'Thus, there is no over-fitting in the train and test dataset')

# create pipeline
pipeline_rf = Pipeline([('scaler1', MinMaxScaler()), ('classifier1', RandomForestClassifier())])
pipeline_logistic = Pipeline([('scaler1', MinMaxScaler()), ('classifier1', LogisticRegression())])
pipeline_xgb = Pipeline([('scaler1', MinMaxScaler()), ('classifier1', XGBClassifier())])

# Create a list of the pipeline
pipelines = [pipeline_rf, pipeline_logistic, pipeline_xgb]

# Create a dictionary of Pipelines for ease of reference 
pipeline_dict = {0: 'Random Forest', 1: 'Logistic Regression', 2: 'XGBoost'}

best_accuracy = 0.0
best_classifier = 0
best_pipeline = ""

for pipe in pipelines:
    pipe.fit(x_train, y_train)

for i, model in enumerate(pipelines):
    print(f'\n{pipeline_dict[i]} Training Accuracy: {model.score(x_train, y_train)}')

for i, model in enumerate(pipelines):
    print(f'\n{pipeline_dict[i]} Test Accuracy: {model.score(x_test, y_test)}')

for i, models in enumerate(pipelines):
    if models.score(x_test, y_test) > best_accuracy:
        best_accuracy = models.score(x_test, y_test)
        best_pipeline = model
        best_classifier = i
print(f'Classifier with the best accuracy: {pipeline_dict[best_classifier]}')

# save your model
# joblib.dump(xgb_model, 'house_classification_model.pkl')
# joblib.dump(rfm, 'home_classification_model.pkl')
joblib.dump(pipeline_rf, 'home_pipeline_model.pkl')

# ------------------- STREAMLIT DEPLOYMENT ----------------------

# Design the landing page
st.markdown("<h1 style = 'text-align: left; color: #57C5B6'>House Classification Model</h1>",unsafe_allow_html=True)
landing_image = Image.open('image/pngwing.com.png')
st.image(landing_image)

# Design the input/interactive page
user_name = st.text_input('Please enter your name')
gender = st.radio('What is your gender?',['No selection','Male','Female'])
# submit = st.button('Submit')
if user_name == '' or gender == 'No selection':
    st.warning('Please input the necessary details above')
elif user_name != '' and gender != 'No selection':
    st.success(f'Welcome, {user_name}! Now, let us help you classify your Perth property using our very efficient model!')

    # Design the sidebar
    # profile picture
    # create a file uploader
    profile = st.sidebar.selectbox('How would you like your profile picture displayed?', ['Placeholder image', 'Choice image'])
    if profile == 'Placeholder image':
        if gender == 'Male':
            male_image = Image.open('image/male.png')
            st.sidebar.image(male_image, use_column_width='auto')
        elif gender == 'Female':
            female_image = Image.open('image/female.png')
            st.sidebar.image(female_image, use_column_width='auto', )
    
    elif profile == 'Choice image':
        profile_mode = st.sidebar.selectbox('Would you like to: ', ['Upload image', 'Take a photo'])
        if profile_mode == 'Upload image':
            profile_picture = st.sidebar.file_uploader('Upload an image')
            if profile_picture is not None:
                profile_picture = Image.open(io.BytesIO(profile_picture.read()))
                st.sidebar.image(profile_picture, caption='My profile picture', use_column_width=True)
            else:
                st.warning('Please upload a profile picture or use placeholder image')

        elif profile_mode == 'Take a photo':
            capture = st.sidebar.button('Click here to take a photo')
            if capture:
                camera = st.sidebar.camera_input('Take your best photo!')
                if camera:
                    save_image = Image.open(io.BytesIO(camera.read()))
                    # camera = Image.open(camera)
                    st.sidebar.image(save_image, caption='My profile picture', use_column_width=True)
                else:
                    st.warning('Please take a photo or use placeholder image')

    st.sidebar.write(f'Hey, {user_name}!')

    # mode of entry
    mode_of_entry = st.sidebar.selectbox('How would you like to input your details today?', ['No selection', 'Individually', 'As a Batch'])
    if mode_of_entry == 'No selection':
        st.sidebar.write(f'Please select a valid option.')
    if mode_of_entry == 'Individually':
        st.sidebar.write('Please input the values for the following features: ')
        
        FLOOR_AREA = st.sidebar.number_input('Please input the floor area of your house (typically between 1 to 870 square meters): ')
        
        NEAREST_SCH_RANK = st.sidebar.number_input('What is the rank of ATAR-applicable school nearest to your house?: ', help='ATAR means Australian Tertiary Admission Rank. It is between the range of 1 to 139. You can refer to the following link for reference: https://bettereducation.com.au/Results/WA/wace.aspx')

        # bathroom selection
        bathroom_list = range(0, 17)
        BATHROOMS = st.sidebar.selectbox('How many bathrooms does the house have?: ', bathroom_list)
        if BATHROOMS == 0:
            st.sidebar.warning(f'Your house has {BATHROOMS} bathrooms')
        if BATHROOMS == 1:
            st.sidebar.success(f'Your house has {BATHROOMS} bathroom')
        if BATHROOMS > 1:
            st.sidebar.success(f'Your house has {BATHROOMS} bathrooms')

        CBD_DIST = st.sidebar.number_input('What is the distance of your house to the Central Business District? (between 681 to 59,800 in kms): ')
        
        # bedroom selection
        room_list = range(0, 11)
        BEDROOMS = st.sidebar.selectbox('How many bedrooms does your house have?: ', room_list)
        if BEDROOMS == 0:
            st.sidebar.warning(f'Your house has {BEDROOMS} rooms')
        if BEDROOMS == 1:
            st.sidebar.success(f'Your house has {BEDROOMS} room')
        if BEDROOMS > 1:
            st.sidebar.success(f'Your house has {BEDROOMS} rooms')

        # build_year selection
        year_list = range(1868, 2018) # ----- Create a list containing the start year and end year for your model
        BUILD_YEAR = st.sidebar.selectbox('What year was the house built?: ', year_list) # ---- Display the list of years as a dropdown for the user
        if BUILD_YEAR > 0:
            st.sidebar.success(f'You selected the year: {BUILD_YEAR}') # ---- Display the user's selected year
        
        # garage selection
        garage_list = range(0, 100)
        GARAGE = st.sidebar.selectbox('How many garages do you have in your house?: ', garage_list)
        if GARAGE == 0:
            st.sidebar.warning(f'Your house has {GARAGE} garages')
        if GARAGE == 1:
            st.sidebar.success(f'Your house has {GARAGE} garage')
        if GARAGE > 1:
            st.sidebar.success(f'Your house has {GARAGE} garages')

        POSTCODE = st.sidebar.number_input('Please enter the postcode of the suburb your house is located in: ', help='This is a four-digit code for the suburb in which your dream house is located, between between 6003 to 6558')

        NEAREST_STN_DIST = st.sidebar.number_input('What is the approximate distance of your house to the nearest train station?: ', help='This is typically between 46 to 35500kms')

        LAND_AREA = st.sidebar.number_input('Please enter the land area of your house (typically between 61 to 999,999 in square meters): ')
    
        predict_button = st.sidebar.button('Classify my property')

        # input_variables = [[floor_area, nearest_sch_rank, bathrooms, cbd_distribution, bedrooms, build_year, garage, postcode, nearest_stn_dist, land_area]]

        input_variables = [[FLOOR_AREA, NEAREST_SCH_RANK, BATHROOMS, CBD_DIST, BEDROOMS, BUILD_YEAR, GARAGE, POSTCODE, NEAREST_STN_DIST, LAND_AREA]]
        input_v = np.array(input_variables)

        frame = ({'FLOOR_AREA':[FLOOR_AREA], 'NEAREST_SCH_RANK': [NEAREST_SCH_RANK], 'BATHROOMS': [BATHROOMS], 'CBD_DIST': [CBD_DIST], 'BEDROOMS': [BEDROOMS], 'BUILD_YEAR': [BUILD_YEAR], 'GARAGE': [GARAGE], 'POSTCODE': [POSTCODE], 'NEAREST_STN_DIST': [NEAREST_STN_DIST], 'LAND_AREA': [LAND_AREA]})

        # st.markdown("<hr><hr>", unsafe_allow_html= True)

        # display user's input information
        st.write('These are your input variables: ')
        frame = pd.DataFrame(frame)
        frame = frame.rename(index = {0: 'Value'})
        frame = frame.transpose()
        st.write(frame)

        # load the model
        model_ = joblib.load(open('home_pipeline_model.pkl','rb'))
        classifier = model_.predict(input_v)
        score = model_.predict_proba(input_v)
        current_date = date.today()

        if predict_button:
            # 'low_cost', 'affordable', 'expensive'
            if classifier == 0:
                st.success('Your dream house is classified as: low_cost')
                st.text(f"Probability Score: {score}")
                st.image('https://images.unsplash.com/photo-1465301055284-72f355cfd745?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MTN8fGxvdyUyMGNvc3QlMjBob3VzZXxlbnwwfHwwfHw%3D&auto=format&fit=crop&w=900&q=60', caption = 'Source: Unsplash', use_column_width = True)
                st.write('These are typically homes between AUD0 to AUD500,000. \n A low cost house has the same foundation, structure, and strength as any other building when it comes to the design and construction of the house. The fact that it is termed a low-cost housing does not mean inferior or poor quality materials were used in constructing it. \n Essentially, these are houses that are relatively inexpensive compared to the average price in the area. They may be small in size or located in less desirable areas. They are typically affordable for low-income earners or first-time homebuyers.. \n', 'Source: https://www.grin.com/document/1290171')
                st.info(f'Classified as at: {current_date}')

            elif classifier == 1:
                st.success('Your dream house is classified as: affordable')
                st.text(f"Probability Score: {score}")
                st.image('https://images.unsplash.com/photo-1572120360610-d971b9d7767c?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2070&q=80', caption = 'Source: Unsplash', use_column_width = True)
                st.write('These are typically homes between AUD500,001 to AUD700,000. \n These are houses that are priced within the means of the average household in the area. They may have moderate-sized rooms and be located in a safe and convenient neighborhood. They are typically suitable for middle-income earners who want to own a home without stretching their budget too much.')
                st.info(f'classified as at: {current_date}')
        
            elif classifier == 2:
                st.warning('Your dream house is classified as: expensive')
                st.text(f"Probability Score: {score}")
                st.image('https://images.unsplash.com/photo-1568605114967-8130f3a36994?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2070&q=80', caption = 'Source: Unsplash', use_column_width = True)
                st.write('These are typically homes from AUD700,001 and above. \n These are houses that are priced significantly higher than the average price in the area. They may have spacious rooms, luxurious finishes, and be located in prestigious neighborhoods. They are typically suitable for high-income earners or those who want to invest in high-end properties.')
                st.info(f'Classified at: {current_date}')

    elif mode_of_entry == 'As a Batch':
        # download file template
        download_file = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vRysnDuksi3sWOB0U1C3Io5i8t9hCw5hVu8yySxP3Bxc7J2nLy4SawXIw9b0XmYd20PtZx59Sbklhl8/pub?output=csv'
        download_button =st.sidebar.download_button('Download a template file', download_file)
        
        # upload populated template file
        upload_file = st.sidebar.file_uploader('Upload your populated template file in csv format')

        # display uploaded data
        file = pd.DataFrame(upload_file)
        st.dataframe(file)


# sel_features = perth[['FLOOR_AREA','NEAREST_SCH_RANK','BATHROOMS','CBD_DIST','BEDROOMS','BUILD_YEAR','GARAGE','POSTCODE','NEAREST_STN_DIST','LAND_AREA']]