import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import google.generativeai as genai
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
import joblib
import ast

# Load training data for encoding
@st.cache_data
def load_training_data():
    df = pd.read_csv("Carbon Emission.csv")
    # Handle complex columns
    df['Vehicle Type'] = df['Vehicle Type'].fillna("unknown")
    for col in ['Recycling', 'Cooking_With']:
        df[col] = df[col].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])
        df[col] = df[col].apply(lambda x: x if len(x) > 0 else ['None'])
    return df

# Load the training data
df = load_training_data()

# Page Configuration
st.set_page_config(
    page_title="Carbon Emission Tracker",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stButton > button {
        width: 100%;
        background-color: #00cc00;
        color: white;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize Gemini AI
@st.cache_resource
def initialize_gemini():
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
        if not api_key:
            st.warning("Gemini API key not found in secrets. Please configure it.")
            return None
            
        genai.configure(api_key=api_key)
        
        # Use the latest stable model
        model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
        
        # Test the model with a simple prompt to verify it works
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40
        }
        
        response = model.generate_content(
            "Hello",
            generation_config=generation_config
        )
        return model
    except Exception as e:
        st.error(f"Failed to initialize Gemini AI: {str(e)}")
        st.warning("AI recommendations will not be available. Please check your API key configuration.")
        return None

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load('carbon_emission_model.joblib')

# Main Content
st.title("🌱 Carbon Emission Tracker")
st.subheader("Powered by AI & Machine Learning")

# Sidebar for main inputs
with st.sidebar:
    st.title("🌍 Personal Information")
    
    # Basic Information
    body_type = st.selectbox("Body Type", 
        ["underweight", "normal", "overweight", "obese"])
    sex = st.selectbox("Sex", ["female", "male"])
    diet = st.selectbox("Diet", 
        ["vegan", "vegetarian", "pescatarian", "omnivore"])
    
    shower_frequency = st.selectbox("How Often Shower", 
        ["less frequently", "daily", "more frequently", "twice a day"])

    # Social and Travel
    social_activity = st.selectbox("Social Activity", 
        ["never", "sometimes", "often"])
    air_travel = st.selectbox("Frequency of Traveling by Air",
        ["never", "rarely", "frequently", "very frequently"])
    
    st.title("🏠 Living & Transport")
    # Energy and Transport
    heating_source = st.selectbox("Heating Energy Source", 
        ["coal", "natural gas", "wood", "electricity"])
    
    transport = st.selectbox("Transport", 
        ["public", "private", "walk/bicycle"])
    
    if transport == "private":
        vehicle_type = st.selectbox("Vehicle Type", 
            ["petrol", "diesel", "electric", "hybrid", "lpg"])
    else:
        vehicle_type = ""

# Main area with additional inputs
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 Monthly Statistics")
    monthly_grocery = st.number_input("Monthly Grocery Bill", 
        min_value=50, max_value=300, value=150,
        help="Your monthly spending on groceries")
    
    vehicle_distance = st.number_input("Vehicle Monthly Distance (Km)", 
        min_value=0, max_value=10000, value=500,
        help="Distance traveled by vehicle per month in kilometers")
    
    new_clothes = st.number_input("How Many New Clothes Monthly", 
        min_value=0, max_value=50, value=5,
        help="Number of new clothing items purchased monthly")
    
    st.markdown("### 🗑️ Waste Management")
    waste_bag_size = st.selectbox("Waste Bag Size", 
        ["small", "medium", "large", "extra large"])
    
    waste_bag_count = st.number_input("Waste Bag Weekly Count", 
        min_value=1, max_value=7, value=3,
        help="Number of waste bags disposed per week")
    
    st.markdown("### ⚡ Energy Usage")
    tv_pc_hours = st.number_input("How Long TV PC Daily Hour", 
        min_value=0, max_value=24, value=4,
        help="Hours spent on TV/PC daily")

with col2:
    st.markdown("### 💻 Digital Usage")
    internet_hours = st.number_input("How Long Internet Daily Hour", 
        min_value=0, max_value=24, value=6,
        help="Hours spent on internet daily")
    
    st.markdown("### 🔋 Energy Efficiency")
    energy_efficiency = st.selectbox("Energy Efficiency", 
        ["No", "Sometimes", "Yes"],
        help="Do you use energy-efficient appliances and practices?")
    
    st.markdown("### 🍳 Cooking & Recycling")
    cooking_methods = st.multiselect("Cooking With", 
        ["Stove", "Oven", "Microwave", "Grill", "Airfryer", "None"],
        default=["Stove"])
    
    recycling_items = st.multiselect("Recycling", 
        ["Paper", "Plastic", "Glass", "Metal", "None"],
        default=["Paper"])

# Calculate Button and Results
st.markdown("### 🎯 Calculate Your Carbon Footprint")
st.markdown('<div class="metric-card">', unsafe_allow_html=True)

if st.button("Calculate Emission"):
    try:
        model = load_model()
        
        # Create input data dictionary
        input_data = {
            # Categorical features
            "Body Type": body_type,
            "Sex": sex,
            "Diet": diet,
            "How Often Shower": shower_frequency,
            "Heating Energy Source": heating_source,
            "Transport": transport,
            "Vehicle Type": vehicle_type if transport == "private" else "",
            "Social Activity": social_activity,
            "Frequency of Traveling by Air": air_travel,
            "Waste Bag Size": waste_bag_size,
            "Energy efficiency": energy_efficiency,

            # Multi-choice features
            "Cooking_With": cooking_methods,
            "Recycling": recycling_items,
            
            # Numerical features
            "Monthly Grocery Bill": monthly_grocery,
            "Vehicle Monthly Distance Km": vehicle_distance,
            "Waste Bag Weekly Count": waste_bag_count,
            "How Long TV PC Daily Hour": tv_pc_hours,
            "How Many New Clothes Monthly": new_clothes,
            "How Long Internet Daily Hour": internet_hours
        }
        
        # Create DataFrame with user input
        input_df = pd.DataFrame([input_data])
        
        # Label encode categorical variables
        label_encode_cols = [
            "Body Type", "Sex", "Diet", "Heating Energy Source",
            "Transport", "Vehicle Type", "Social Activity",
            "Frequency of Traveling by Air", "Waste Bag Size", "Energy efficiency"
        ]
        
        encoded_df = input_df.copy()
        le = LabelEncoder()
        
        for col in label_encode_cols:
            # Fit on training data categories + new input
            combined_data = pd.concat([df[col], pd.Series(input_data[col])])
            le.fit(combined_data)
            encoded_df[col + "_encoded"] = le.transform([input_data[col]])
        
        # Encode shower frequency
        shower_mapping = {"less frequently": 0, "daily": 1, "more frequently": 2, "twice a day": 3}
        encoded_df["HowOftenShower_encoded"] = shower_mapping[input_data["How Often Shower"]]
        
        # MultiLabel encode Cooking_With and Recycling
        mlb = MultiLabelBinarizer()
        
        for col in ["Cooking_With", "Recycling"]:
            # Fit on all possible categories from training data
            mlb.fit(df[col])
            encoded = pd.DataFrame(
                mlb.transform([input_data[col]]), 
                columns=[f"{col}_{c}" for c in mlb.classes_],
                index=encoded_df.index
            )
            encoded_df = pd.concat([encoded_df, encoded], axis=1)
        
        # Drop original categorical columns
        drop_cols = label_encode_cols + ["Cooking_With", "Recycling", "How Often Shower"]
        encoded_df = encoded_df.drop(columns=drop_cols)
        
        # Use the encoded dataframe for prediction
        X = encoded_df
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        # Display results
        st.metric("Estimated Annual Carbon Emission",
            f"{prediction:.2f} kg CO₂e")
        
        # Compare with averages
        avg_emission = 2500  # Global average
        percentage = ((prediction - avg_emission) / avg_emission) * 100
        
        if percentage < 0:
            st.success(f"🌟 Your carbon footprint is {abs(percentage):.1f}% below average!")
        else:
            st.warning(f"⚠️ Your carbon footprint is {percentage:.1f}% above average.")
        
        # Show breakdown
        st.markdown("### 📊 Emission Breakdown")
        categories = {
            'Transport': prediction * 0.35,
            'Home Energy': prediction * 0.25,
            'Food & Groceries': prediction * 0.20,
            'Waste': prediction * 0.12,
            'Other Activities': prediction * 0.08
        }
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(categories.keys()),
                y=list(categories.values()),
                marker_color=['#2ecc71', '#3498db', '#e74c3c', '#f1c40f', '#95a5a6']
            )
        ])
        
        fig.update_layout(
            title='Carbon Emission by Category',
            xaxis_title='Category',
            yaxis_title='Emission (kg CO₂e)',
            template='plotly_white'
        )
        
        st.plotly_chart(fig)
        
        # Get AI Recommendations
        model_ai = initialize_gemini()
        if model_ai is not None:
            try:
                prompt = f"""
                Based on the following lifestyle choices:
                - Diet: {diet}
                - Transport: {transport} {f'({vehicle_type})' if vehicle_type else ''}
                - Heating: {heating_source}
                - Shower Frequency: {shower_frequency}
                - Recycling Habits: {', '.join(recycling_items)}
                - Cooking Methods: {', '.join(cooking_methods)}
                - Energy Efficiency: {energy_efficiency}
                
                The user's carbon footprint is {prediction:.2f} kg CO₂e per year, which is {percentage:.1f}% {"above" if percentage > 0 else "below"} average.
                
                Provide 3 specific, actionable recommendations to reduce their carbon emissions.
                Format your response with bullet points and emojis.
                Make the suggestions practical and tailored to their current habits.
                """
                
                generation_config = {
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 40
                }
                
                response = model_ai.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                
                st.markdown("### 🤖 AI-Powered Recommendations")
                st.markdown(response.text)
            except Exception as e:
                st.error("Could not generate AI recommendations at this time.")
                st.info("Here are some general tips to reduce carbon emissions:")
                st.markdown("""
                * 🚶‍♂️ Consider using public transport or walking for short distances
                * 💡 Switch to energy-efficient appliances and LED bulbs
                * ♻️ Increase recycling and reduce single-use items
                """)
        else:
            st.info("AI recommendations are currently unavailable.")
            st.markdown("""
            ### 🌱 General Tips for Reducing Carbon Emissions:
            * 🚶‍♂️ Consider using public transport or walking for short distances
            * 💡 Switch to energy-efficient appliances and LED bulbs
            * ♻️ Increase recycling and reduce single-use items
            """)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please make sure all fields are filled correctly.")

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("### 💡 About This Tool")
st.write("""
This Carbon Emission Tracker uses machine learning to estimate your carbon footprint based on your lifestyle choices.
The predictions are enhanced with Google's Gemini AI to provide personalized recommendations for reducing your environmental impact.
""")

