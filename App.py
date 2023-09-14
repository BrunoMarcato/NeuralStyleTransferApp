import streamlit as st
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub

# Carregando o modelo pré-treinado
model_path = "models\magenta_arbitrary-image-stylization-v1-256_2"
model = hub.load(model_path)

# Função para realizar a transferência de estilo
# Função para realizar a transferência de estilo
#@st.cache_resource
def style_transfer(_content_image, _style_image):
    # Redimensionar as imagens para o tamanho esperado pelo modelo (256x256)
    size = 256
    content_image = _content_image.resize((size, size))
    style_image = _style_image.resize((size, size))

    content_image = tf.image.convert_image_dtype(tf.convert_to_tensor(content_image), tf.float32)
    style_image = tf.image.convert_image_dtype(tf.convert_to_tensor(style_image), tf.float32)

    # Expandir dimensão para criar um lote com tamanho 1
    content_image = tf.expand_dims(content_image, axis=0)
    style_image = tf.expand_dims(style_image, axis=0)

    stylized_image = model(content_image, style_image)[0]
    return stylized_image.numpy()

# Título do aplicativo
st.title("Style Transfer App")

# Upload das imagens de conteúdo e estilo
content_image = st.file_uploader("Carregue a imagem de conteúdo", type=["jpg", "jpeg", "png"])
style_image = st.file_uploader("Carregue a imagem de estilo", type=["jpg", "jpeg", "png"])

if content_image is not None and style_image is not None:
    # Carregando as imagens
    content_image = Image.open(content_image)
    style_image = Image.open(style_image)

    # Realizando a transferência de estilo
    stylized_result = style_transfer(content_image, style_image)

    # Exibindo a imagem resultante
    st.subheader("Imagem Resultante:")
    st.image(stylized_result, use_column_width=True, channels="RGB")
