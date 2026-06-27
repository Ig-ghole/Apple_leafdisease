import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

import cv2


# Page configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title="Apple Leaf Disease Detection",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Header styling */
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    
    /* Subheader styling */
    .sub-header {
        font-size: 1.5rem;
        color: #1B5E20;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Card styling for results */
    .result-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 5px solid #2E7D32;
    }
    
    .result-card-healthy {
        background: linear-gradient(135deg, #e8f5e9 0%, #a5d6a7 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 5px solid #2E7D32;
    }
    
    .result-card-disease {
        background: linear-gradient(135deg, #ffebee 0%, #ef9a9a 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 5px solid #c62828;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #2E7D32 0%, #1B5E20 100%);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        background: linear-gradient(135deg, #1B5E20 0%, #0d3d13 100%);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Image container */
    .image-container {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    /* Info box styling */
    .info-box {
        background-color: #e3f2fd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1565C0;
        margin: 1rem 0;
    }
    
    /* Feature cards */
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: transform 0.3s ease;
        height: 100%;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)


# Add these functions after your import section and before load_model()

def check_image_quality(image):
    """Check if image contains valid leaf content"""
    try:
        # Convert to numpy array if needed
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
            
        # Check image dimensions
        if img_array.shape[0] < 50 or img_array.shape[1] < 50:
            return False, "Image is too small (minimum 50x50 pixels)"
        
        # Check for extremely dark or bright images
        mean_brightness = np.mean(img_array)
        if mean_brightness < 30:
            return False, "Image is too dark. Please use better lighting."
        if mean_brightness > 230:
            return False, "Image is too bright or overexposed."
        
        # Check for uniform images (like solid colors or noise)
        std_brightness = np.std(img_array)
        if std_brightness < 15:
            return False, "Image appears to be uniform (solid color or noise)"
        
        # Check if image has green color (common in leaves)
        if len(img_array.shape) == 3:
            green_channel = img_array[:, :, 1]
            avg_green = np.mean(green_channel)
            if avg_green < 30:
                return False, "Image doesn't appear to contain green leaves"
        
        # Check for blurriness using Laplacian variance
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 50:  # Threshold for blur detection
            return False, "Image is too blurry. Please take a clearer photo."
        
        return True, "Image quality is acceptable"
        
    except Exception as e:
        return False, f"Error checking image: {e}"

def predict_with_confidence(test_image, model, confidence_threshold=0.70):
    """Enhanced prediction with confidence threshold"""
    try:
        # Preprocess the image
        image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])
        
        # Get predictions
        predictions = model.predict(input_arr, verbose=0)
        confidence = float(np.max(predictions) * 100)  # Convert to float here
        result_index = int(np.argmax(predictions))     # Convert to int here
        confidence = float(confidence) / 100 
        # Check if confidence is above threshold
        if confidence >= confidence_threshold * 100:
            return result_index, confidence, True  # Valid prediction
        else:
            return result_index, confidence, False  # Low confidence - likely invalid image
            
    except Exception as e:
        st.error(f"Error predicting disease: {e}")
        return None, None, False

# Tensorflow model prediction
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('trained_plant_disease_model.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


# Load model once
model = load_model()

# Define class names
class_names = ['Apple Scab', 'Black Rot', 'Cedar Apple Rust', 'Healthy']
disease_info = {
    'Apple Scab': {
        'description': 'Apple scab is a common disease caused by the fungus Venturia inaequalis.',
        'treatment': 'Apply fungicides, remove fallen leaves, and prune infected branches.',
        'severity': 'Moderate to High'
    },
    'Black Rot': {
        'description': 'Black rot is caused by the fungus Botryosphaeria obtusa.',
        'treatment': 'Remove infected fruit and branches, apply fungicides, ensure proper sanitation.',
        'severity': 'High'
    },
    'Cedar Apple Rust': {
        'description': 'Cedar-apple rust is caused by the fungus Gymnosporangium juniperi-virginianae.',
        'treatment': 'Remove cedar trees nearby, apply fungicides, plant resistant varieties.',
        'severity': 'Moderate'
    },
    'Healthy': {
        'description': 'Your apple leaf appears to be healthy with no signs of disease.',
        'treatment': 'Continue regular maintenance and monitoring.',
        'severity': 'None'
    }
}

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4134/4134160.png", width=80)
    st.title("🍎 Leaf Dashboard")
    st.markdown("---")
    app_mode = st.selectbox(
        "Choose page",
        ["🏠 Home", "ℹ️ About", "🔍 Predict Disease"],
        format_func=lambda x: x.split(" ")[1] if " " in x else x
    )
    st.markdown("---")
    st.caption("Made with ❤️ for farmers and agriculture enthusiasts")

# Home page
if app_mode == "🏠 Home":
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<p class="main-header">🍎 Apple Leaf Disease Detection</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">AI-Powered Agriculture Solution</p>', unsafe_allow_html=True)
    
    # Features section
    st.markdown("### 🌟 Key Features")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📸</div>
            <h4>Easy Upload</h4>
            <p>Simply upload an image of any apple leaf</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🤖</div>
            <h4>AI Analysis</h4>
            <p>Advanced machine learning for accurate detection</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <h4>Fast Results</h4>
            <p>Get results in seconds with confidence scores</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Display image
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        try:
            st.image("home.jpg", caption='🍎 Apple Leaf Disease Detection', use_column_width=True)
        except:
            st.info("ℹ️ Please add a 'home.jpg' image to display here")
    
    # Welcome message
    st.markdown("---")
    st.markdown("""
    ### 🌿 Welcome to the Apple Leaf Disease Recognition System!
    
    Our mission is to help identify apple leaf diseases efficiently. Upload an image, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!
    
    #### 📋 How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page
    2. **Analysis:** Our AI analyzes the leaf for diseases
    3. **Results:** Get instant diagnosis and treatment recommendations
    
    #### 🚀 Get Started
    Click on the **🔍 Predict Disease** page in the sidebar to begin!
    """)
    
    st.info("💡 **Tip:** For best results, upload clear, well-lit images of the entire leaf.")

# About page
elif app_mode == "ℹ️ About":
    st.markdown('<p class="main-header">ℹ️ About the System</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 📊 Dataset Information
        This dataset is recreated using offline augmentation from the original dataset available on Kaggle.
        
        **Dataset Statistics:**
        - **Training:** 658 images
        - **Validation:** 291 images
        - **Testing:** 1 image
        
        **Disease Classes:**
        - 🍂 Apple Scab
        - 🍂 Black Rot
        - 🍂 Cedar Apple Rust
        - ✅ Healthy
        
        [🔗 View Dataset on Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
        """)
    
    with col2:
        st.markdown("""
        ### 🧠 Technology Stack
        - **Framework:** TensorFlow / Keras
        - **Frontend:** Streamlit
        - **Model:** Convolutional Neural Network
        - **Image Processing:** PIL, NumPy
        
        ### 🎯 Project Goals
        1. Provide accurate disease detection
        2. Help farmers make informed decisions
        3. Reduce crop loss through early detection
        4. Promote sustainable agriculture
        """)
    
    st.markdown("---")
    st.markdown("""
    ### 👨‍🌾 Why This Matters
    
    Apple leaf diseases can significantly impact crop yield and quality. Early detection and proper treatment are crucial for maintaining healthy orchards. Our AI-powered system helps farmers and gardeners quickly identify diseases and take appropriate action.
    
    ### 🔬 How It Works
    
    The system uses a deep learning model trained on thousands of labeled images of apple leaves. When you upload an image, the model:
    1. Processes the image
    2. Extracts key features
    3. Compares with learned patterns
    4. Provides the most likely diagnosis
    """)
    
    st.success("🌱 **Together, we can build a more sustainable future for agriculture!**")

# Predict Disease page
elif app_mode == "🔍 Predict Disease":
    st.markdown('<p class="main-header">🔍 Disease Recognition</p>', unsafe_allow_html=True)
    
    # Create two columns for upload and display
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Image")
        test_image = st.file_uploader(
            "Choose an image of an apple leaf",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG"
        )
        
        if test_image is not None:
            st.markdown("---")
            if st.button("🔍 Predict Disease", use_container_width=True):
                if model is None:
                    st.error("❌ Model not loaded. Please check the model file.")
                else:
                    with st.spinner("Analyzing the leaf... Please wait..."):
                        time.sleep(1)  # Simulate processing
                        result_index, confidence, is_valid = predict_with_confidence(test_image, model, confidence_threshold=0.70)

                        if not is_valid:
                            st.error("This is not a valid leaf image 🌿")
                        else:
                            disease_name = class_names[result_index]
                            is_healthy = disease_name == "Healthy"

                            st.success(disease_name)
                            st.write(f"Confidence: {confidence:.2f}%")
                            
                            # Display results
                            st.markdown("---")
                            st.markdown("### 📊 Results")
                            
                            # Custom card based on health status
                            card_class = "result-card-healthy" if is_healthy else "result-card-disease"
                            st.markdown(f"""
                            <div class="{card_class}">
                                <h3>{'✅' if is_healthy else '⚠️'} {disease_name}</h3>
                                <p><strong>Confidence:</strong> {confidence:.1f}%</p>
                                <p><strong>Severity:</strong> {disease_info[disease_name]['severity']}</p>
                                <hr>
                                <p><strong>Description:</strong> {disease_info[disease_name]['description']}</p>
                                <p><strong>Recommended Treatment:</strong> {disease_info[disease_name]['treatment']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Additional recommendations
                            if not is_healthy:
                                st.warning("⚠️ **Action Required:** Consult with an agricultural expert for proper treatment implementation.")
                            else:
                                st.success("🌿 **Great news!** Your plant appears healthy. Continue regular care and monitoring.")
                            
                            # Display confidence bar
                            st.markdown("### 📈 Confidence Level")
                            st.progress(confidence/100)
    
    with col2:
        if test_image is not None:
            st.markdown("### 🖼️ Uploaded Image")
            try:
                image = Image.open(test_image)
                st.image(image, caption='Apple Leaf Sample', use_column_width=True)
            except:
                st.error("Error loading image. Please try uploading again.")
            
            st.markdown("---")
            st.markdown("### 📋 Tips for Best Results")
            st.info("""
            - 📸 Use clear, well-lit images
            - 🍃 Capture the entire leaf
            - 🎯 Ensure the leaf fills most of the frame
            - 🌿 Take photos from straight above
            """)
        else:
            st.markdown("### 📸 No Image Uploaded")
            st.markdown("""
            <div style="text-align: center; padding: 3rem; background-color: #f5f5f5; border-radius: 10px;">
                <div style="font-size: 5rem;">📤</div>
                <h3>Upload an image to begin</h3>
                <p>Click the "Browse files" button to select an image</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.caption("🔬 Powered by Deep Learning | 🌱 For healthier apple crops")

# Run with: streamlit run main.py