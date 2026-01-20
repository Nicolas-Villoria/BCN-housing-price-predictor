import streamlit as st
import os
import sys

# Add project root to path so we can import from 'app'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Page config must be first
st.set_page_config(
    page_title="Lil Homey | BCN",
    page_icon="👓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize Spark immediately (Cached)
with st.spinner("Waking up Lil Homey..."):
    from app.services.model_service import init_spark, load_model, predict_price
    spark = init_spark()
    model = load_model()

from app.components.ui import render_header, render_result_card
from app.components.forms import render_input_form

def main():
    # Render layout
    render_header()
    
    st.markdown("---")
    
    col_left, col_right = st.columns([1, 1.5], gap="large")
    
    with col_left:
        # Input Form
        features = render_input_form()
    
    with col_right:
        # Welcome / Results Area
        st.markdown("### Market Intelligence")
        
        if features is None:
            st.info("👈 Fill out the property details on the left (or top on mobile) to get an instant valuation.")
            
            st.markdown("""
            <div style="background-color: #f8fafc; padding: 20px; border-radius: 10px; border: 1px solid #e2e8f0; margin-top: 20px;">
                <h4 style="margin-top:0;">Why trust Lil Homey?</h4>
                <p style="font-size: 0.95rem; color: #475569;">
                    We don't just guess numbers. We analyze <b>neighborhood wealth indices</b> and <b>population density</b> alongside thousands of real listings to give you a valuation that reflects the <i>true vibe</i> of the area.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            if spark is None:
                st.error("Spark engine failed to initialize. Check logs.")
            elif model is None:
                st.error("Could not load trained model. Ensure 'mlruns' directory exists.")
            else:
                # Perform Inference
                with st.spinner("Crunching the numbers..."):
                    prediction = predict_price(spark, model, features)
                    
                # Display Result
                render_result_card(prediction)
                
                # Explanation Section
                st.markdown("#### What drove this price?")
                st.markdown(f"""
                This valuation is heavily influenced by the **{features['neighborhood']}** location profile:
                - **Wealth Factor:** {features['avg_income_index']:.2f} (Index)
                - **Density Factor:** {features['density_val']:.2f} (Pop/High)
                
                *A higher wealth factor typically drives prices up, while extreme density can have variable effects.*
                """)

if __name__ == "__main__":
    main()

