import streamlit as st
import os
import sys

# Add project root to path so we can import from 'app'
# This fixes ModuleNotFoundError when running 'streamlit run app/main.py'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Page config must be first
st.set_page_config(
    page_title="Barcelona Property Valuator",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Spark immediately (Cached)
with st.spinner("Initializing AI Engine... (First load may take 10s)"):
    from app.services.model_service import init_spark, load_model, predict_price
    spark = init_spark()
    model = load_model()

from app.components.ui import render_header, render_result_card
from app.components.forms import render_input_form

def main():
    # Render layout
    render_header()
    
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        # Input Form
        features = render_input_form()
    
    with col_right:
        # Welcome / Results Area
        if features is None:
            st.info("👈 Please configure the property details in the form to get a valuation.")
            
            # Show global explanations or hero image here
            if os.path.exists("reports/feature_importance.png"):
                st.image("reports/feature_importance.png", caption="Global Feature Importance (What drives prices?)")
            
        else:
            if spark is None:
                st.error("Spark engine failed to initialize. Check logs.")
            elif model is None:
                st.error("Could not load trained model. Ensure 'mlruns' directory exists and contains a valid model.")
            else:
                # Perform Inference
                with st.spinner("Calculating market value..."):
                    prediction = predict_price(spark, model, features)
                    
                # Display Result
                render_result_card(prediction)
                
                # Explanation Section
                with st.expander("Why this price?", expanded=True):
                    st.write(f"The model estimated **{prediction:,.0f} €** based on:")
                    st.markdown(f"""
                    - **Size:** {features['size']} m²
                    - **Location:** {features['neighborhood']} ({features['district']})
                    - **Income Index:** {features['avg_income_index']:.2f} (Neighborhood wealth factor)
                    - **Density:** {features['density_val']:.2f} (Neighborhood density factor)
                    """)
                    
                    if features['avg_income_index'] == 0 or features['density_val'] == 0:
                        st.warning("⚠️ Using default/zero values for Income or Density. "
                                   "This may affect accuracy. Ensure 'data_lake/silver' files exist.")

if __name__ == "__main__":
    main()
