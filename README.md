# 🛠️ How to Set Up & Run the Project

### ✅ 1. Create and Activate a Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it (macOS/Linux)
source venv/bin/activate

# On Windows
venv\\Scripts\\activate

#install required libraries
pip install -r requirements.txt
```

<B> Wind Speed Prediction Using VAE-S2S-BiLSTM-Enc-Dec and Streamlit </B>
<B> Objective </B>
This project aims to predict the wind speed for the next 5 days using a deep learning approach that combines a Variational Autoencoder (VAE) with a sequence-to-sequence Bidirectional LSTM  Encoder-decoder (VAE-S2S-BiLSTM-Enc-Dec) network. The final solution is wrapped into an interactive, user-friendly web application using Streamlit.

<B>1. Data Preparation </B>

<B> Dataset: </B> DailyBeijingClimate.csv, containing historical weather records with features: meantemp, humidity, wind_speed, meanpressure.
- Converted date to datetime and set it as index.
- Selected relevant features and applied MinMaxScaler for normalization.
- Created time sequences using a sliding window with:
  
  • n_past = 10 (past days used for input)
  
  • n_future = 5 (days predicted)
  
- Split data into training and test sets (80:20).

<B> 2. Training and Evaluation </B>

- Trained the model for 200 epochs with batch size 16.
- Evaluation metrics included: MAE, MSE, RMSE, R², and Normalized MSE.
- Scalers for each feature were saved using joblib for consistent inference.

<B>  Deployment: Interactive Streamlit App </B>
An interactive web dashboard was built using Streamlit for real-time wind speed forecasting.
Key Features:
- Accepts user input for today’s weather (meantemp, humidity, wind_speed, meanpressure).
- Applies trained scalers and model inference.
- Displays:
  
  • Predicted wind speed for the next 5 days
  
  • Weather-level icons (🍃, 🌬️, 🌪️)
  
  • Average wind speed visual indicator
  
- Uses custom CSS styling (glassmorphism, gradients, hover effects) for a polished UI.
- Supports session state to persist predictions.
- Ready for deployment or local use.
  
<B>  3. Results </B>  
- The app successfully integrates a real, trained deep learning model, not random predictions.
- The system generalizes well to unseen inputs and offers predictions in a user-friendly and visually rich format.
- The entire pipeline — from data preprocessing to model inference to web visualization — is automated and reusable.

![App](https://github.com/user-attachments/assets/1c94bfb3-b376-4b7f-b698-0045ccf3f0ac)

<B> 4. Tools & Technologies </B>  
- Python (TensorFlow, NumPy, Pandas, Scikit-learn, Joblib)
- Deep Learning: VAE + BiLSTM
- Frontend: Streamlit with custom HTML/CSS
- Deployment-ready: App can be run locally or hosted on Streamlit Cloud
  
<B>  5. Deliverables </B>  
- Trained model: training.h5
- Scalers: scaler_*.save (one per feature)
- Streamlit app: app.py
- Optional assets: logo.png, bg_image.jpg
