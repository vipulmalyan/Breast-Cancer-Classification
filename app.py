import streamlit as st
import numpy as np
import joblib
import pickle 

st.set_page_config(layout="wide")

import pickle

# Load the saved model using pickle
with open('bc.pkl', 'rb') as file:
    model = pickle.load(file)


def predict(input_data):
    # Reshape the input data as we are predicting for one data point
    input_data_reshaped = np.asarray(input_data).reshape(1, -1)
    prediction = model.predict(input_data_reshaped)
    return prediction[0]

def main():
    st.title('Breast Cancer Prediction')

    st.write('Please adjust the values for prediction:')
    
    # Create side sliders for user input
    input_data = {}
    columns = ['Mean Radius', 'Mean Texture', 'Mean Perimeter', 'Mean Area', 'Mean Smoothness', 
               'Mean Compactness', 'Mean Concavity', 'Mean Concave Points', 'Mean Symmetry', 
               'Mean Fractal Dimension', 'Radius Error', 'Texture Error', 'Perimeter Error', 
               'Area Error', 'Smoothness Error', 'Compactness Error', 'Concavity Error', 
               'Concave Points Error', 'Symmetry Error', 'Fractal Dimension Error', 
               'Worst Radius', 'Worst Texture', 'Worst Perimeter', 'Worst Area', 
               'Worst Smoothness', 'Worst Compactness', 'Worst Concavity', 'Worst Concave Points', 
               'Worst Symmetry', 'Worst Fractal Dimension']

    for col in columns:
        input_data[col] = st.sidebar.slider(col, min_value=0.0, max_value=200.0, value=0.0, step=0.1)

    if st.button('Predict'):
        input_values = [value for value in input_data.values()]
        prediction = predict(input_values)
        if prediction == 0:
            st.write('The Breast cancer is Malignant')
        else:
            st.write('The Breast Cancer is Benign')

if __name__ == '__main__':
    main()

    

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