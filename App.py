import streamlit as st
import cv2
import torch
import tensorflow_hub as hub

import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

utils = utils.Utils(style_images_dir='style_images', models_dir='models', images_dir='content_images')

st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: #86895D;
    }
</style>
""", unsafe_allow_html=True)

st.title("Neural Style Transfer")

st.sidebar.title("Settings")
st.sidebar.header("Model")
method = st.sidebar.radio('Select an option', options=['Specific', 'Arbitrary'])

if method == 'Specific':
    style_model_name = st.sidebar.selectbox("Choose the style model: ", utils.formated_names(utils.models))
    model = utils.get_model_from_name(style_model_name)
    style_image = utils.get_image_from_name(style_model_name, style=True)
    
    st.sidebar.image(style_image, width=300)

    if st.sidebar.checkbox('Upload'):
        content_file = st.sidebar.file_uploader("Upload a Content Image", type=["png", "jpg", "jpeg"])

        if content_file is not None:
            content = utils.get_image_from_file(content_file)
        else:
            st.warning("Upload an image")
            st.stop()

    else:
        content_name = st.sidebar.selectbox("Choose a Content Image", utils.formated_names(utils.images))
        content = utils.get_image_from_name(content_name)
    
    style = None

else:
    model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

    content_file = st.sidebar.file_uploader("Upload a Content Image", type=["png", "jpg", "jpeg"])
    style_file = st.sidebar.file_uploader("Upload a Style Image", type=["png", "jpg", "jpeg"])

    if content_file is not None and style_file is not None:
            content = utils.get_image_from_file(content_file)
            style = utils.get_image_from_file(style_file)
    else:
        st.warning("Upload content and style images")
        st.stop()

WIDTH = st.sidebar.select_slider('RESIZE', list(range(150, 1001, 50)), value=300)
content = cv2.resize(content, (WIDTH, WIDTH))

st.sidebar.image(content, width=300, channels='RGB')

# apply style transfer
result = utils.stylize(content, model, style, method)

if method == 'Specific':
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    if type(model) != cv2.dnn.Net:
        result = result.astype('long')

    st.image(result, clamp=True, width=700)
else:
    st.image(result.numpy(), clamp=True, width=700)