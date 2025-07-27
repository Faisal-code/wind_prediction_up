import streamlit as st
import numpy as np
import base64
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

# ----- Register Sampling Function -----
@tf.keras.utils.register_keras_serializable()
def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# ----- Custom KL Divergence Layer -----
class KLDivergenceLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(KLDivergenceLayer, self).__init__(**kwargs)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        self.add_loss(kl_loss)
        return z_mean

# ----- Constants -----
features = ['meantemp', 'humidity', 'wind_speed', 'meanpressure']
n_past = 10
n_future = 5
n_features = len(features)

# ----- Load Model and Scalers -----
model = load_model(
    "vae_bilstm_model.h5",
    custom_objects={"KLDivergenceLayer": KLDivergenceLayer, "sampling": sampling}
)
scalers = {f'scaler_{f}': joblib.load(f'scaler_{f}.save') for f in features}

# ----- UI Styling -----
def get_base64(bin_file):
    with open(bin_file, "rb") as file:
        return base64.b64encode(file.read()).decode()

def set_background(image_file):
    bin_str = get_base64(image_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)), url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    .main {{
        background-color: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        padding: 2rem;
    }}
    .input-box {{
        background-color: rgba(255, 255, 255, 0.2);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }}
    .prediction-card {{
        background: linear-gradient(135deg, #6e8efb, #a777e3);
        border-radius: 10px;
        padding: 1.5rem;
        color: white;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        margin-bottom: 1rem;
    }}
    .header {{
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }}
    .stButton>button {{
        background: linear-gradient(135deg, #6e8efb, #a777e3);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-size: 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
    }}
    .stButton>button:hover {{
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
    }}
    .day-card {{
        background: rgba(255, 255, 255, 0.15);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid #6e8efb;
    }}
    .footer {{
        text-align: center;
        color: white;
        opacity: 0.7;
        margin-top: 3rem;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# ----- Streamlit Page Config -----
st.set_page_config(
    page_title="Wind Speed Prediction", 
    page_icon="🌬️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply background and styles
set_background("bg_image.jpg")

# ----- Header Section -----
st.markdown("""
<div class="header">
    <h1 style="font-size: 2.5rem; margin-bottom: 0.5rem;">🌬️ WindSpeed Predictor</h1>
    <p style="font-size: 1.1rem; opacity: 0.9;">Predict the next 5 days of wind speed using advanced AI models</p>
</div>
""", unsafe_allow_html=True)

# ----- Main Content -----
with st.container():
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("""
        <div class="input-box">
            <h3 style="color: white; margin-bottom: 1.5rem;">📊 Input Weather Data</h3>
        """, unsafe_allow_html=True)
        
        # User Inputs with icons
        meantemp = st.number_input("🌡️ Mean Temperature (°C)", min_value=-50.0, max_value=60.0, value=25.0)
        humidity = st.number_input("💧 Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
        wind_speed = st.number_input("🍃 Current Wind Speed (km/h)", min_value=0.0, value=10.0)
        meanpressure = st.number_input("⏱️ Mean Pressure (millibar)", min_value=800.0, max_value=1100.0, value=1013.0)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Prediction Button
        if st.button("🚀 Predict Wind Speed", key="predict"):
            with st.spinner('🔮 Predicting future wind patterns...'):
                # Scale today's input
                today_scaled = np.array([
                    scalers['scaler_meantemp'].transform([[meantemp]])[0][0],
                    scalers['scaler_humidity'].transform([[humidity]])[0][0],
                    scalers['scaler_wind_speed'].transform([[wind_speed]])[0][0],
                    scalers['scaler_meanpressure'].transform([[meanpressure]])[0][0]
                ])

                # Create dummy past data (zeros) + today's scaled input
                dummy_history = np.zeros((n_past - 1, n_features))
                sequence = np.vstack([dummy_history, today_scaled])
                input_array = sequence.reshape(1, n_past, n_features)

                # Model prediction
                prediction = model.predict(input_array)  # shape: (1, 5, 4)

                # Inverse transform the wind_speed only (index 2)
                wind_speed_scaled = prediction[0, :, 2].reshape(-1, 1)
                wind_speed_pred = scalers['scaler_wind_speed'].inverse_transform(wind_speed_scaled)
                
                # Store predictions in session state
                st.session_state.predictions = wind_speed_pred
    
    with col2:
        st.markdown("""
        <div style="background-color: rgba(255, 255, 255, 0.1); border-radius: 10px; padding: 1.5rem;">
            <h3 style="color: white; margin-bottom: 1.5rem;">📈 Prediction Results</h3>
        """, unsafe_allow_html=True)
        
        if 'predictions' in st.session_state:
            st.markdown("""
            <div class="prediction-card">
                <h4 style="margin-top: 0; margin-bottom: 1rem;">🌪️ Predicted Wind Speeds</h4>
            """, unsafe_allow_html=True)
            
            for i, speed in enumerate(st.session_state.predictions, start=1):
                # Determine icon based on wind speed
                if speed[0] < 10:
                    icon = "🍃"
                elif 10 <= speed[0] < 20:
                    icon = "🌬️"
                else:
                    icon = "🌪️"
                
                st.markdown(f"""
                <div class="day-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span><b>Day {i}:</b></span>
                        <span>{icon} <b>{speed[0]:.2f} km/h</b></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Visual indicator
            avg_speed = np.mean(st.session_state.predictions)
            st.markdown(f"""
            <div style="margin-top: 1.5rem;">
                <p style="color: white; margin-bottom: 0.5rem;">Average predicted wind speed:</p>
                <div style="background: rgba(255, 255, 255, 0.2); border-radius: 10px; height: 10px;">
                    <div style="background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%); 
                                width: {min(100, avg_speed/30*100)}%; 
                                height: 100%; 
                                border-radius: 10px;"></div>
                </div>
                <p style="color: white; text-align: right; margin-top: 0.5rem;">{avg_speed:.2f} km/h</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: rgba(255, 255, 255, 0.1); border-radius: 10px; padding: 2rem; text-align: center;">
                <p style="color: white; opacity: 0.8;">👈 Enter weather data and click the predict button to see forecast</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

# ----- Footer -----
st.markdown("""
<div class="footer">
    <p></p>
</div>
""", unsafe_allow_html=True)