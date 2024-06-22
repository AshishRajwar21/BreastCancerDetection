
#import  important libraries
import numpy as np
import pandas as pd
import streamlit as st
import pickle

#load the dataset
df = pd.read_csv('data/data.csv')

#headings
st.title('Breast Cancer Classification')
st.sidebar.header('Patient Data')
st.subheader('Training Data Statistics')
st.write(df.describe())


#function to take input from slider and convert it into dataframe
def patientReport():
  radius_mean = st.sidebar.slider('Radius Mean', 0.0,df['radius_mean'].max(), df['radius_mean'].mean())
  texture_mean = st.sidebar.slider('Texture Mean', 0.0,df['texture_mean'].max(), df['texture_mean'].mean())
  perimeter_mean = st.sidebar.slider('Perimeter Mean', 0.0,df['perimeter_mean'].max(), df['perimeter_mean'].mean())
  area_mean = st.sidebar.slider('Area Mean', 0.0,df['area_mean'].max(), df['area_mean'].mean())
  smoothness_mean = st.sidebar.slider('Smoothness Mean', 0.0,df['smoothness_mean'].max(), df['smoothness_mean'].mean())
  compactness_mean = st.sidebar.slider('Compactness Mean', 0.0,df['compactness_mean'].max(), df['compactness_mean'].mean())
  concavity_mean = st.sidebar.slider('Concavity Mean', 0.0,df['concavity_mean'].max(), df['concavity_mean'].mean())
  concave_points_mean = st.sidebar.slider('Concave Points Mean', 0.0,df['concave points_mean'].max(), df['concave points_mean'].mean())
  symmetry_mean = st.sidebar.slider('Symmetry Mean', 0.0,df['symmetry_mean'].max(), df['symmetry_mean'].mean())
  fractal_dimension_mean = st.sidebar.slider('Fractal Dimension Mean', 0.0,df['fractal_dimension_mean'].max(), df['fractal_dimension_mean'].mean())
  radius_se = st.sidebar.slider('Radius SE', 0.0,df['radius_se'].max(), df['radius_se'].mean())
  texture_se = st.sidebar.slider('Texture SE', 0.0,df['texture_se'].max(), df['texture_se'].mean())
  perimeter_se = st.sidebar.slider('Perimeter SE', 0.0,df['perimeter_se'].max(), df['perimeter_se'].mean())
  area_se = st.sidebar.slider('Area SE', 0.0,df['area_se'].max(), df['area_se'].mean())
  smoothness_se = st.sidebar.slider('Smoothness SE', 0.0,df['smoothness_se'].max(), df['smoothness_se'].mean())
  compactness_se = st.sidebar.slider('Compactness SE', 0.0,df['compactness_se'].max(), df['compactness_se'].mean())
  concavity_se = st.sidebar.slider('Concavity SE', 0.0,df['concavity_se'].max(), df['concavity_se'].mean())
  concave_points_se = st.sidebar.slider('Concave Points SE', 0.0,df['concave points_se'].max(), df['concave points_se'].mean())
  symmetry_se = st.sidebar.slider('Symmetry SE', 0.0,df['symmetry_se'].max(), df['symmetry_se'].mean())
  fractal_dimension_se = st.sidebar.slider('Fractal Dimension SE', 0.0,df['fractal_dimension_se'].max(), df['fractal_dimension_se'].mean())
  radius_worst = st.sidebar.slider('Radius Worst', 0.0,df['radius_worst'].max(), df['radius_worst'].mean())
  texture_worst = st.sidebar.slider('Texture Worst', 0.0,df['texture_worst'].max(), df['texture_worst'].mean())
  perimeter_worst = st.sidebar.slider('Perimeter Worst', 0.0,df['perimeter_worst'].max(), df['perimeter_worst'].mean())
  area_worst = st.sidebar.slider('Area Worst', 0.0,df['area_worst'].max(), df['area_worst'].mean())
  smoothness_worst = st.sidebar.slider('Smoothness Worst', 0.0,df['smoothness_worst'].max(), df['smoothness_worst'].mean())
  compactness_worst = st.sidebar.slider('Compactness Worst', 0.0,df['compactness_worst'].max(), df['compactness_worst'].mean())
  concavity_worst = st.sidebar.slider('Concavity Worst', 0.0,df['concavity_worst'].max(), df['concavity_worst'].mean())
  concave_points_worst = st.sidebar.slider('Concave Points Worst', 0.0,df['concave points_worst'].max(), df['concave points_worst'].mean())
  symmetry_worst = st.sidebar.slider('Symmetry Worst', 0.0,df['symmetry_worst'].max(), df['symmetry_worst'].mean())
  fractal_dimension_worst = st.sidebar.slider('Fractal Dimension Worst', 0.0,df['fractal_dimension_worst'].max(), df['fractal_dimension_worst'].mean())
  patient_report_data = {
    'radius_mean': radius_mean,
    'texture_mean': texture_mean,
    'perimeter_mean': perimeter_mean, 
    'area_mean': area_mean,
    'smoothness_mean': smoothness_mean,
    'compactness_mean': compactness_mean,
    'concavity_mean': concavity_mean,
    'concave points_mean': concave_points_mean,
    'symmetry_mean': symmetry_mean,
    'fractal_dimension_mean': fractal_dimension_mean,
    'radius_se': radius_se,
    'texture_se': texture_se,
    'perimeter_se': perimeter_se,
    'area_se': area_se,
    'smoothness_se': smoothness_se,
    'compactness_se': compactness_se,
    'concavity_se': concavity_se,
    'concave points_se': concave_points_se,
    'symmetry_se': symmetry_se,
    'fractal_dimension_se': fractal_dimension_se,
    'radius_worst': radius_worst,
    'texture_worst': texture_worst,
    'perimeter_worst': perimeter_worst,
    'area_worst': area_worst,
    'smoothness_worst': smoothness_worst,
    'compactness_worst': compactness_worst,
    'concavity_worst': concavity_worst,
    'concave points_worst': concave_points_worst,
    'symmetry_worst': symmetry_worst,
    'fractal_dimension_worst': fractal_dimension_worst
  }
  report_data = pd.DataFrame(patient_report_data, index=[0])
  return report_data


#displaying the patient data
user_data = patientReport()


st.subheader('Patient Report Data')
st.write(user_data)

#load the model data using pickle
model_data = pickle.load(open('model/model.pkl','rb'))

#standardization of data
scaler = model_data['scaler']

#standardize the user data
user_data_standardization = scaler.transform(user_data)

#make prediction on user data
model = model_data['model']
prediction = model.predict(user_data_standardization)

prediction_label = [np.argmax(prediction,axis=1)]

#Final prediction
st.subheader('Your Report: ')
result=''
if prediction_label[0]==0:
  result = 'The tumor is Malignant.'
else:
  result = 'The tumor is Benign.'
st.subheader(result)
#find the accuracy of the model
st.subheader('Accuracy: ')
accuracy = model_data['accuracy']
st.write(str(accuracy*100)+'%')

