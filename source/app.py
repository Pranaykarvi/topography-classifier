import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import base64

# Function to add a custom button for GitHub and LinkedIn
def add_custom_button(label, url, icon_url):
    st.markdown(
        f"""
        <a href="{url}" target="_blank">
            <button style="
                background-color: #4CAF50; 
                border: none; 
                color: white; 
                padding: 10px 20px; 
                text-align: center; 
                text-decoration: none; 
                display: inline-block; 
                font-size: 16px; 
                margin: 4px 2px; 
                cursor: pointer;
                border-radius: 12px;">
                <img src="{icon_url}" alt="{label}" style="vertical-align: middle; width:20px; margin-right:8px;">
                {label}
            </button>
        </a>
        """, 
        unsafe_allow_html=True,
    )

# Function to preprocess the uploaded image
def preprocess_image(img):
    # Ensure image is RGB (3 channels)
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    # Resize to model input size
    img = img.resize((224, 224))
    
    # Convert to numpy array and normalize
    img = np.array(img) / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    return img

# Function to load the pre-trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('source/my_model.keras')
    return model

# Load the model only once
model = load_model()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Home", "Dataset", "About Me", "Prediction"])

# Home Page
if page == "Home":
    st.title("🌍 Topography Classification")
    # Display project image
    st.image("source/aerial-view-vouglan-dam-reservoir-north-oyonnax-france.jpg", 
             caption="Topography Classification", use_column_width=True)
    st.write("""
    Welcome to the **Topography Classifier App**!  
    This application uses a Deep Learning model to classify satellite images into four categories:
    - **Sky/Cloud**
    - **Water**
    - **Desert**
    - **Forest**
    
    ### Problem Statement
    Satellites capture a vast amount of data, and classifying these images into meaningful categories is a challenge. 
    This app demonstrates how machine learning can address this problem efficiently.
    
    ### Approach
    The model uses a **custom ensemble of DenseNet, ResNet, and Xception architectures**, which are powerful for image classification tasks. 
    It was trained on satellite images to achieve high accuracy in topography classification.
    
    ### Techniques Used
    - Transfer Learning (DenseNet, ResNet, Xception)
    - Image Augmentation for better generalization
    - Categorical Cross-Entropy Loss and Adam Optimizer
    - Evaluation Metrics: Accuracy and Loss
    """)

# Dataset Page
elif page == "Dataset":
    st.title("📊 About the Dataset")
    st.write("""
    The dataset used for this project contains satellite images representing different types of topography.  
    Images are categorized into four classes:
    - **Sky/Cloud**
    - **Water**
    - **Desert**
    - **Forest**
    
    Each image was resized to 224x224 pixels for training. The dataset was split into training, validation, and testing subsets to ensure reliable evaluation.
    """)

# About Me Page
elif page == "About Me":
    st.title("👨‍💻 About Me")
    st.write("""
    **Hello!**  
    I’m a budding Data Scientist in my second year at VIT Chennai, deeply engaged in unraveling the complexities of data to find patterns that others might overlook.  
    With a robust technical foundation in languages like Python, C++, and specialized tools for machine learning and data analysis, I am prepared to tackle diverse datasets with precision.

    My certifications from esteemed institutions and active involvement in tech forums reflect my commitment to continuous learning and excellence.  
    Curiosity drives my exploration of emerging tech, aiming to stay at the forefront of innovation.

    I’m on the lookout for opportunities to grow and contribute to transformative projects in data science.  
    **Want to see what we can discover together?** Connect with me here:
    """)

    # Add GitHub and LinkedIn buttons
    add_custom_button("GitHub", "https://github.com/Pranaykarvi", 
                      "https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg")
    add_custom_button("LinkedIn", "https://www.linkedin.com/in/pranaykarvi/", 
                      "https://cdn-icons-png.flaticon.com/512/174/174857.png")

# Prediction Page
elif page == "Prediction":
    st.title("🖼️ Make a Prediction")
    st.write("Upload a satellite image to classify its topographical category.")

    # Upload Image
    uploaded_file = st.file_uploader("Upload an Image File", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Make prediction
        if st.button("Classify Image"):
            with st.spinner("Classifying..."):
                try:
                    prediction = model.predict(processed_image)
                    predicted_class = np.argmax(prediction)
                    class_labels = ["Sky/Cloud", "Desert", "Forest", "Water"]
                    st.success(f"Prediction: **{class_labels[predicted_class]}**")
                except Exception as e:
                    st.error(f"Error during prediction: {e}")

# Footer
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        text-align: center;
        font-size: 30px;
        background-color: rgba(0, 0, 0, 0.7);
        color: white;
        padding: 10px 0;
    }
    </style>
    <div class="footer">
        © 2024 Pranay Karvi. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)
