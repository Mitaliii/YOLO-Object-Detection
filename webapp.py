#Import libraries
import streamlit as st
import numpy as np
import os
import cv2
from  PIL import Image, ImageEnhance
from yolov4 import yolo_detection
import matplotlib.pyplot as plt

# image = Image.open(r'giraffe.jpg') #Brand logo image (optional)

#Create two columns with different width
# col1, col2 = st.columns( [0.9, 0.1])
# with col1:               # To display the header text using css style
st.markdown(""" <style> .font {
font-size:25px ; font-family: 'Cooper Black'; color: #ffa31a;} 
</style> """, unsafe_allow_html=True)
st.markdown('<p class="font">Upload your photo to detect objects</p>', unsafe_allow_html=True)
    
# with col2:               # To display brand logo
#     st.image(image,  width=350)

#Add a header and expander in side bar
st.sidebar.markdown('<p class="font">Object Detection App</p>', unsafe_allow_html=True)
with st.sidebar.expander("About the App"):
     st.write("""
        A user-friendly webapp to detect objects in a given image. \n  \nThis app was created as a side project to learn Streamlit, computer vision and YOLO algorithm. Hope you enjoy!
     """)



uploaded_file = st.file_uploader("", type=['jpg'])
folder = 'C:\\Users\\User\\Downloads\\'
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_path = uploaded_file.name
    
    col1, col2 = st.columns( [0.5, 0.5])
    with col1:
        st.markdown('<p style="text-align: center;">Original Image</p>',unsafe_allow_html=True)
        st.image(image,use_column_width=True)  
        # st.markdown(img_path)
    od_image,objs_dict = yolo_detection(os.path.join(folder,img_path))
    with col2:
        labels = objs_dict.keys()
        sizes = objs_dict.values()
        st.markdown('<p style="text-align: center;">Ratio of Total Objects</p>',unsafe_allow_html=True)        
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, labels=labels, autopct='%1.0f%%', startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig1)



    st.markdown('<p style="text-align: center;">Objects Detected in image</p>',unsafe_allow_html=True)
    converted_img = cv2.cvtColor(od_image, cv2.COLOR_RGB2BGR)
    st.image(converted_img,use_column_width=True)

        # filter = st.sidebar.radio('Covert your photo to:', ['Original','Gray Image','Black and White', 'Pencil Sketch', 'Blur Effect'])
        # if filter == 'Gray Image':
        #         converted_img = np.array(image.convert('RGB'))
        #         gray_scale = cv2.cvtColor(converted_img, cv2.COLOR_RGB2GRAY)
        #         st.image(gray_scale, width=300)
        # elif filter == 'Black and White':
        #         converted_img = np.array(image.convert('RGB'))
        #         gray_scale = cv2.cvtColor(converted_img, cv2.COLOR_RGB2GRAY)
        #         slider = st.sidebar.slider('Adjust the intensity', 1, 255, 127, step=1)
        #         (thresh, blackAndWhiteImage) = cv2.threshold(gray_scale, slider, 255, cv2.THRESH_BINARY)
        #         st.image(blackAndWhiteImage, width=300)
        # elif filter == 'Pencil Sketch':
        #         converted_img = np.array(image.convert('RGB')) 
        #         gray_scale = cv2.cvtColor(converted_img, cv2.COLOR_RGB2GRAY)
        #         inv_gray = 255 - gray_scale
        #         slider = st.sidebar.slider('Adjust the intensity', 25, 255, 125, step=2)
        #         blur_image = cv2.GaussianBlur(inv_gray, (slider,slider), 0, 0)
        #         sketch = cv2.divide(gray_scale, 255 - blur_image, scale=256)
        #         st.image(sketch, width=300) 
        # elif filter == 'Blur Effect':
        #         converted_img = np.array(image.convert('RGB'))
        #         slider = st.sidebar.slider('Adjust the intensity', 5, 81, 33, step=2)
        #         converted_img = cv2.cvtColor(converted_img, cv2.COLOR_RGB2BGR)
        #         blur_image = cv2.GaussianBlur(converted_img, (slider,slider), 0, 0)
        #         st.image(blur_image, channels='BGR', width=300) 
        # else: 
        #         st.image(image, width=300)