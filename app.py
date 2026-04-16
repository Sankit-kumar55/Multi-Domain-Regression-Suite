import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Multi-Model Prediction Suite",
    page_icon="🤖",
    layout="wide"
)

# --- ATTRACTIVE GUI STYLING ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border: none;
    }
    .prediction-container {
        padding: 30px;
        border-radius: 15px;
        background-color: #ffffff;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        text-align: center;
        margin-top: 20px;
    }
    .price-display {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 10px 0;
    }
    .footer { text-align: center; padding: 20px; color: #666; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🛠️ Control Panel")
st.sidebar.markdown("Select the model you wish to use for prediction.")
app_mode = st.sidebar.selectbox("Choose a Model",
                                ["Salary Prediction 💼", "Device Price Prediction 📱", "House Price Prediction 🏠",
                                 "House Rent Prediction 🔑"])

# --- 1. SALARY PREDICTION ---
if "Salary Prediction" in app_mode:
    st.title("💼 Salary Prediction System")
    st.info("Input your professional details below to estimate your annual salary.")

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Select Age", 18, 65, 30, help="User's current age")
        gender = st.selectbox("Gender", ["Male", "Female"])
        # Mapping: Male=1, Female=0
        gender_enc = 1 if gender == "Male" else 0

        education = st.selectbox("Education Level", ["Bachelor", "High School", "Master", "PhD"])
        # Mapping: Bachelor=0, High School=1, Master=2, PhD=3
        edu_map = {"Bachelor": 0, "High School": 1, "Master": 2, "PhD": 3}

    with col2:
        experience = st.slider("Years of Experience", 0, 40, 5, help="Total professional experience")
        job_title = st.selectbox("Job Title", ["Analyst", "Director", "Engineer", "Manager"])
        # Mapping: Analyst=0, Director=1, Engineer=2, Manager=3
        job_map = {"Analyst": 0, "Director": 1, "Engineer": 2, "Manager": 3}

        location = st.selectbox("Location Type", ["Rural", "Suburban", "Urban"])
        # Mapping: Rural=0, Suburban=1, Urban=2
        loc_map = {"Rural": 0, "Suburban": 1, "Urban": 2}

    if st.button("Calculate Expected Salary"):
        try:
            model = joblib.load('salary_xgb_model.joblib')
            # Feature order: Education, Experience, Location, Job_Title, Age, Gender
            features = np.array([[edu_map[education], experience, loc_map[location],
                                  job_map[job_title], age, gender_enc]])
            prediction = model.predict(features)[0]

            st.markdown(f"""
                <div class="prediction-container">
                    <h3>Estimated Annual Salary</h3>
                    <p class="price-display" style="color: #2E7D32;">₹ {prediction:,.2f}</p>
                    <p>(Currency: Indian Rupees)</p>
                </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error: Ensure 'salary_xgb_model.joblib' is in your directory. ({e})")

# --- 2. DEVICE PRICE PREDICTION ---
elif "Device Price" in app_mode:
    st.title("📱 Device Price Predictor")
    st.info("Estimate the resale market value of a mobile device.")

    c1, c2, c3 = st.columns(3)
    with c1:
        brand_map = {"Others" : 24 , "Samsung" : 27 , "Huawei" : 11 , "LG" : 14 , "Lenovo" : 16 ,  "ZTE" : 33 , "Xiaomi" : 32 , "Oppo" : 23 , "Asus" : 3, "Alcatel" : 10 , "Micromax" : 30 , "Vivo" : 18 ,"Honor" : 9 ,"HTC" : 1 , 
                                                        "Nokia" : 20 ,"Motorola" : 21 ,"Sony" : 17 ,"Meizu" : 7 ,"Gionee" : 28 ,"Acer" : 0 ,"XOLO" : 31 ,"Panasonic" : 25 ,"Realme" : 26 ,"Apple" : 2 ,"Lava" : 29 ,"Celkon" : 15 ,"Spice" : 13 , 
                                                        "Karbonn" : 22 ,"BlackBerry" : 6 ,"OnePlus" : 19 ,"Microsoft" : 4 ,"Coolpad" : 8 ,"Google" : 5 , "Infinix" : 12 }       
        brand = st.selectbox("Device Brand ", list(brand_map.keys())) #[Others , Samsung , Huawei , LG ,Lenovo ,  ZTE , Xiaomi , Oppo , Asus , Alcatel , Micromax , Vivo  ,Honor ,HTC  , 
                                                       # Nokia ,Motorola  ,Sony ,Meizu  ,Gionee  ,Acer  ,XOLO  ,Panasonic  ,Realme  ,Apple  ,Lava  ,Celkon  ,Spice  , 
                                                       # Karbonn  ,BlackBerry ,OnePlus ,Microsoft ,Coolpad ,Google ,Infinix ])     
        os_map = {"Android": 0, "Others": 1, "Windows": 2, "iOS": 3}  #
        os = st.selectbox("Operating System", list(os_map.keys()))
        screen_size = st.slider("SCREEN SIZE OF THE DEVICE " ,5 , 31 , 14 , help = " CM ")
        year = st.selectbox("Release Year", [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020 , 2021 , 2022 , 2023 , 2024 ])

    with c2:
        ram = st.selectbox("RAM (GB)", [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0])
        internal = st.selectbox("Internal Memory (GB)", [8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0])
        battery = st.selectbox("Battery Capacity (mAh)", [2000.0, 3000.0, 4000.0, 5000.0])
        weight = st.slider("Weight of the device" , 70 , 855 , 140 , help = " weight in Gram ")

    with c3:
        rear = st.selectbox("Rear Camera (MP)", [5.0, 8.0, 12.0, 13.0, 16.0, 48.0])
        front = st.selectbox("Front Camera (MP)", [2.0, 5.0, 8.0, 16.0, 32.0])
        g4 = 1 if st.selectbox("4G Supported?", ["yes", "no"]) == "yes" else 0
        g5 = 1 if st.selectbox("5G Supported?", ["yes", "no"]) == "yes" else 0
        days = st.slider("Days device used", 365 , 1865 , 365 , help = " enter the number of days device is used ")

    actual_new = st.number_input("Original New Price ($)", min_value=0.0, value=500.0)

    if st.button("Predict Resale Price"):
        try:
            model = joblib.load('device_price.joblib')
            # Features: [brand, os, screen, 4g, 5g, rear, front, memory, ram, battery, weight, year, days, new_price]
            features = np.array(
                [[brand_map[brand], os_map[os], screen_size, g4, g5, rear, front, internal, ram, battery, weight, year, days , actual_new]])
            prediction = model.predict(features)[0]

            st.markdown(f"""
                <div class="prediction-container">
                    <h3>Predicted Resale Value</h3>
                    <p class="price-display" style="color: #1565C0;">$ {prediction:,.2f}</p>
                    <p>(Currency: US Dollars)</p>
                </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Model Error: {e}")

# --- 3. HOUSE PRICE PREDICTION ---
elif "House Price Prediction" in app_mode:
    st.title("🏠 House Purchase Price Predictor")
    st.info("Predict the total purchase value of a property.")

    col1, col2 = st.columns(2)
    with col1:
        area = st.number_input("Total Area (sq ft)", value=1500)
        bed = st.selectbox("Bedrooms", [1, 2, 3, 4, 5])
        bath = st.selectbox("Bathrooms", [1, 2, 3, 4])
        floor = st.selectbox("Total Floors", [1, 2, 3])
        # House Mappings
        #bed_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
        #bath_map = {1: 0, 2: 1, 3: 2, 4: 3}
        #floor_map = {1: 0, 2: 1, 3: 2}

    with col2:
        year_built = st.slider("Year Built", 1990, 2026, 2018)
        loc_map = {"Downtown": 0, "Rural": 1, "Suburban": 2, "Urban": 3}  #
        location = st.selectbox("Location", list(loc_map.keys()))
        cond_map = {"Excellent": 0, "Fair": 1, "Good": 2, "Poor": 3}  #
        condition = st.selectbox("Property Condition", list(cond_map.keys()))
        garage = 1 if st.selectbox("Garage Attached?", ["Yes", "No"]) == "Yes" else 0

    if st.button("Predict Buying Price"):
        try:
            model = joblib.load('House_Price_Prediction.joblib')
            features = np.array([[area, bed, bath, floor, year_built, loc_map[location],
                                  cond_map[condition], garage]])
            prediction = model.predict(features)[0]

            st.markdown(f"""
                <div class="prediction-container">
                    <h3>Estimated Market Price</h3>
                    <p class="price-display" style="color: #C62828;">₹ {prediction:,.2f}</p>
                    <p>(Currency: Indian Rupees)</p>
                </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"File 'House_Price_Prediction.joblib' not found. ({e})")

# --- 4. HOUSE RENT PREDICTION ---
elif "House Rent Prediction" in app_mode:
    st.title("🔑 House Rent Predictor")
    st.info("Predict the monthly rental cost for a house based on location and specs.")

    col1, col2 = st.columns(2)
    with col1:
        area = st.number_input("Area (sq ft)", value=1000)
        bed = st.selectbox("Bedrooms", [1, 2, 3, 4, 5], key="rent_bed")
        bath = st.selectbox("Bathrooms", [1, 2, 3, 4], key="rent_bath")
        floor = st.selectbox("Floor Number", [1, 2, 3], key="rent_floor")

    with col2:
        loc_map = {"Downtown": 0, "Rural": 1, "Suburban": 2, "Urban": 3}  #
        location = st.selectbox("Location", list(loc_map.keys()), key="rent_loc")
        cond_map = {"Excellent": 0, "Fair": 1, "Good": 2, "Poor": 3}  #
        condition = st.selectbox("Condition", list(cond_map.keys()), key="rent_cond")
        garage = 0 if st.selectbox("Garage Available?", ["Yes", "No"], key="rent_gar") == "Yes" else 1

    if st.button("Predict Monthly Rent"):
        try:
            # Assuming separate model for rent as stated
            model = joblib.load('House_Rent_Prediction.joblib')
            # Encoding mappings applied
            features = np.array(
                [[area, bed - 1, bath - 1, floor - 1, 2024, loc_map[location], cond_map[condition], garage]])
            prediction = model.predict(features)[0]

            st.markdown(f"""
                <div class="prediction-container">
                    <h3>Estimated Monthly Rent</h3>
                    <p class="price-display" style="color: #6A1B9A;">₹ {prediction:,.2f}</p>
                    <p>(Currency: Indian Rupees)</p>
                </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Please upload 'House_Rent_Prediction.joblib' to use this model. ({e})")

st.markdown("---")
st.markdown('<div class="footer">Built with Streamlit & Machine Learning</div>', unsafe_allow_html=True)
