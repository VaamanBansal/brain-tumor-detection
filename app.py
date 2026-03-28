import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

st.title("🧠 Brain Tumor Detection with Grad-CAM")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("/mnt/c/Users/Lenovo/OneDrive/Desktop/Codes/TF/proj_1/tumor_model.keras")
model = load_model()
base = model.layers[0]

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Original Image", width="stretch")

    img = image.resize((224,224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    prediction = model.predict(img_array)[0][0]

    st.subheader("Prediction")

    if prediction > 0.5:
        st.error(f"Tumor Detected")
    else:
        st.success(f"No Tumor")

    st.write(f"Confidence: {prediction:.2f}")
    st.caption("Threshold = 0.5 (above → tumor)")

    with tf.GradientTape() as tape:
        x = img_tensor
        for layer in model.layers:
            x = layer(x)
            if layer == base:
                conv_outputs = x

        preds = x
        loss = preds[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap)

    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    heatmap_resized = cv2.resize(heatmap.numpy(), (img_cv.shape[1], img_cv.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    superimposed_img = heatmap_color * 0.4 + img_cv

    st.subheader("Visualization")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original")

    with col2:
        st.image(
            cv2.cvtColor(superimposed_img.astype('uint8'), cv2.COLOR_BGR2RGB),
            caption="Grad-CAM Overlay"
        )

    st.subheader("Heatmap (Importance Scale)")

    fig, ax = plt.subplots()
    im = ax.imshow(heatmap_resized, cmap='jet', vmin=0, vmax=1)
    ax.axis("off")
    cbar = plt.colorbar(im)
    cbar.set_label("Importance")

    st.pyplot(fig)