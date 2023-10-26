import streamlit as st
import cv2

from utils.utils import Utils
from style_transfer import apply_style_transfer


utils = Utils(models_dir='models', images_dir='images')

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
else:
    style_model_name = 'magenta_arbitrary-image-stylization-v1-256_2'

style_model = utils.get_model_from_name(style_model_name)

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

WIDTH = st.sidebar.select_slider('RESIZE', list(range(150, 501, 50)), value=300)
content = cv2.resize(content, (WIDTH, WIDTH))

st.sidebar.image(content, width=300, channels='RGB')


# apply style transfer
#result = apply_style_transfer(content, style_model)

#show result
#st.image(result, channels='RGB', clamp=True)