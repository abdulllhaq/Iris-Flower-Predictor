#iris flower predictor
#works perfectly

import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from PIL import Image,ImageFilter,ImageEnhance
import os





st.markdown('''
# Iris Flower Species Predictor 
This app detects the type of Iris flower based on Machine Learning!
- App built by Pranav Sawant and Anshuman Shukla of Team Skillocity.
''')
st.write('---')

st.title('Diabetes Detector')
st.sidebar.header('Input Features')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction])
#st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)

# Iris Image Manipulation
@st.cache
def load_image(img):
    im =Image.open(os.path.join(img))
    return im

# Select Image Type using Radio Button
species_type = st.radio('What is the Iris Species do you want to see?',('Setosa','Versicolor','Virginica'))

if species_type == 'Setosa':
    st.text("Showing Setosa Species")
    st.image(load_image('https://github.com/pranav-coder2005/iris_flower_predictor/blob/main/iris_setosa.jpg'))
elif species_type == 'Versicolor':
    st.text("Showing Versicolor Species")
    st.image(load_image('https://github.com/pranav-coder2005/iris_flower_predictor/blob/main/IRIS_VERSICOLOR.jpg'))
elif species_type == 'Virginica':
    st.text("Showing Virginica Species")
    st.image(load_image('https://github.com/pranav-coder2005/iris_flower_predictor/blob/main/virginca.jpg'))
