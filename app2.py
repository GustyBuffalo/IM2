import streamlit as st
import numpy as np
import cv2
from get_image_urls import get_image_urls #get image
from tensorflow.keras.models import Model
import pandas
from PIL import Image
import requests
from sklearn.svm import LinearSVC
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import io

st.markdown("<h1 style='text-align: center; '>Image Classification</h1>", unsafe_allow_html=True)
st.markdown(
    """<style>
div[class*="stRadio"] > label > div[data-testid="stMarkdownContainer"] > p {
    font-size: 32px;
}
    </style>
    """, unsafe_allow_html=True)
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

st.divider()

num = st.radio(
    "How many do you want to compare?",
    ('2', '3', '4','5'))

if num == '2':
  input_1 = st.text_input("Enter some text1 👇", placeholder='สุนัข',)
  input_2 = st.text_input("Enter some text2 👇", placeholder='แมว',)
  if input_1:
    if input_2:
      texts = [input_1,input_2]
elif num == '3':
  input_1 = st.text_input("Enter some text1 👇", placeholder='สุนัข',)
  input_2 = st.text_input("Enter some text2 👇", placeholder='แมว',)
  input_3 = st.text_input("Enter some text3 👇", placeholder='กระต่าย',)
  if input_1:
    if input_2:
      if input_3:
        texts = [input_1,input_2,input_3]
elif num == '4':
  input_1 = st.text_input("Enter some text1 👇", placeholder='สุนัข',)
  input_2 = st.text_input("Enter some text2 👇", placeholder='แมว',)
  input_3 = st.text_input("Enter some text3 👇", placeholder='กระต่าย',)
  input_4 = st.text_input("Enter some text4 👇", placeholder='หนู',)
  if input_1:
    if input_2:
      if input_3:
        if input_4:
          texts = [input_1,input_2,input_3,input_4]
else:
  input_1 = st.text_input("Enter some text1 👇", placeholder='สุนัข',)
  input_2 = st.text_input("Enter some text2 👇", placeholder='แมว',)
  input_3 = st.text_input("Enter some text3 👇", placeholder='กระต่าย',)
  input_4 = st.text_input("Enter some text4 👇", placeholder='หนู',)
  input_5 = st.text_input("Enter some text5 👇", placeholder='งู',)
  if input_1:
    if input_2:
      if input_3:
        if input_4:
          if input_5:
            texts = [input_1,input_2,input_3,input_4,input_5]

st.divider()
st.subheader('How many images do you want to train?')
number = st.slider('Choose between 5-50',
              min_value=5,max_value=50,step=1)
st.write('Images train number:', number)
st.write(' ')
st.divider()
choose = st.radio(
    "URL or Upload from your PC",
    ('URL', 'Upload'))
if choose == 'URL':
  url = st.text_input("Enter url:",placeholder='Type here',)
  uploaded_file = False
else:
  uploaded_file = st.file_uploader("Choose a image")
  url = False

result = st.button('Classify')
st.divider()
if result:
  with st.spinner('Wait for it...'):
    
    def encoded(url):
      response = requests.get(url)
      response.raise_for_status()
      image_data = response.content
      return encode(image_data)

    def encode(image):
      # Convert image data to numpy array
      nparr = np.asarray(bytearray(image), dtype=np.uint8)
      # Read image with OpenCV
      image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

      # Resize the image
      resized_image = cv2.resize(image, (224, 224))
      # Convert the image to an array
      image = img_to_array(resized_image)
      # Expand dimensions to match the shape required by VGG16 (batch size 1)
      image = image.reshape((-1, image.shape[0], image.shape[1], image.shape[2]))
      image = preprocess_input(image)
      # Load the VGG16 model (pre-trained on ImageNet)
      model = VGG16(weights='imagenet', include_top=False)
      # Define the desired intermediate layer for encoding
      intermediate_layer = 'block5_pool'
      # Create a new model with the desired intermediate layer as the output
      model = Model(inputs=model.input, outputs=model.get_layer(intermediate_layer).output)
      # Encode the image using the VGG16 model
      encoded_image = model.predict(image)
      encoded = encoded_image.flatten()
      return encoded

    def featextraction(queryList,n):
      dataList = []
      for query in queryList:
        info = get_image_urls(query)
        featList = []
        for images in info[0:n]:
          try:
            image = encoded(images)
            featList.append(image)
          except requests.exceptions.RequestException as e:
            print(f"Error downloading image from {images}: {str(e)}")

        dat = pandas.DataFrame(data=[featList]).T
        dat['label'] = query
        dat.columns = ['feature','label']
        dataList.append(dat)
      return pandas.concat(dataList)

    def trainmodel(res):
      clf = LinearSVC()
      clf.fit(np.vstack(res['feature'].values),res['label'].values)
      return clf

    def predict(imgurl,clf):
      encode = encoded(imgurl)
      # Remove null bytes from the URL
      url = imgurl.replace('%00', '')  
      st.image(imgurl)
      return clf.predict([encode])[0]

    def upload_img(imgpc,clf):
      # Load and preprocess the image
      st.image(imgpc)
      image_path = imgpc
      # Read the image file
      # Open the image file
      image = Image.open(image_path)
      # Resize the image to fit VGG16 input dimensions (224x224)
      resized_image = image.resize((224, 224))
      # Convert the image to an array
      image = img_to_array(resized_image)
      # Expand dimensions to match the shape required by VGG16 (batch size 1)
      image = image.reshape((-1, image.shape[0], image.shape[1], image.shape[2]))
      image = preprocess_input(image)
      # Load the VGG16 model (pre-trained on ImageNet)
      model = VGG16(weights='imagenet', include_top=False)
      # Define the desired intermediate layer for encoding
      intermediate_layer = 'block5_pool'
      # Create a new model with the desired intermediate layer as the output
      model = Model(inputs=model.input, outputs=model.get_layer(intermediate_layer).output)
      # Encode the image using the VGG16 model
      encoded_image = model.predict(image)
      encoded = encoded_image.flatten()
      return clf.predict([encoded])[0]

    res = featextraction(texts,number)
    model = trainmodel(res)
    with st.expander("See explanation: "):
      st.subheader('Data Frame of images')
      st.write(res)
    
    with st.sidebar:
      if uploaded_file == False:
        st.write('')
      else:
        st.balloons()
        predictPC = upload_img(uploaded_file,model)
        predictPC

      if url == False:
        st.write('')
      else:
        st.balloons()
        Prediction = predict(url, model)
        Prediction
