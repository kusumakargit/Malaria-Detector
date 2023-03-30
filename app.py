#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# load the models
model1 = load_model('malaria_model.h5')
model2 = load_model('Malaria_model_basic_CNN.h5')

# define a function to predict using model 1
def predict_model1(image):
    # preprocess the image
    img = np.array(image.convert('RGB').resize((64, 64))) / 255.0
    img = np.expand_dims(img, axis=0)

    # make a prediction using model 1
    prediction = model1.predict(img)

    return prediction

# define a function to predict using model 2
def predict_model2(image):
    # preprocess the image
    img = np.array(image.convert('RGB').resize((128, 128))) / 255.0
    img = np.expand_dims(img, axis=0)

    # make a prediction using model 2
    prediction = model2.predict(img)

    return prediction

# create the Streamlit app
def app():
    st.title('Malaria Detector - CNN Model Comparison')
    st.write('Upload an image and see the predictions of two CNN models!')

    # create file uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # check if file has been uploaded
    if uploaded_file is not None:
        # load the image
        image = Image.open(uploaded_file)

        # display the image
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # make a prediction using model 1
        prediction1 = predict_model1(image)

        # make a prediction using model 2
        prediction2 = predict_model2(image)

        # display the predictions
        st.subheader('Model 1 Prediction - Transfer Learning_VGG19')
        st.write(prediction1)
        if np.argmax(prediction1) == 0:
            st.write("Malaria Parasite Not Present in the Blood sample")
        else:
            st.write("Malaria Parasite Present in the Blood sample")


        st.subheader('Model 2 Prediction - Built from Scratch')
        st.write(prediction2)
        if np.round(prediction2) == 1:
            st.write("Malaria Parasite Not Present in the Blood sample")
        else:
            st.write("Malaria Parasite Present in the Blood sample")

# run the app
if __name__ == '__main__':
    app()


# In[ ]:




