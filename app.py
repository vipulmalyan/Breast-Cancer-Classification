import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(layout="wide")

# Loading the breast cancer dataset
breast_cancer_dataset = load_breast_cancer()

# Creating a DataFrame
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)
data_frame['label'] = breast_cancer_dataset.target

# Separating features and target
X = data_frame.drop(columns='label', axis=1)
Y = data_frame['label']

# Splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Training the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Streamlit app
st.title('Breast Cancer Classification App')

# Sidebar for user input
st.sidebar.title('Enter New Data for Prediction')
input_data = []
for feature in breast_cancer_dataset.feature_names:
    value = st.sidebar.slider(f'Enter {feature}', float(X[feature].min()), float(X[feature].max()), float(X[feature].mean()))
    input_data.append(value)

# Predict function
def predict(input_data):
    input_data_reshaped = np.asarray(input_data).reshape(1, -1)
    prediction = model.predict(input_data_reshaped)
    return prediction

if st.sidebar.button('Predict'):
    prediction = predict(input_data)
    if prediction[0] == 0:
        st.write('The Breast cancer is Malignant.')
    else:
        st.write('The Breast Cancer is Benign.')


page_bg_img = '''
<style>
[data-testid="stAppViewContainer"] {
background-image: url("https://static.vecteezy.com/system/resources/previews/020/149/566/original/simple-silhouette-women-background-pink-vector.jpg");
background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

hide_st_style = '''
<style> footer {visibility: hidden;} 
</style>
'''
st.markdown(hide_st_style, unsafe_allow_html=True)

footer_html = """
<style>
.footer {
    position: fixed;
    bottom: 0;
    width: 100%;
    color: white; /* Text color */
    padding: 10px;
    text-align: center; /* Center the text */
    font-size: 18px; /* Adjust the font size */
}
</style>
<div class="footer">Made by Vipul Malyan</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)