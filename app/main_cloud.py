"""
Barcelona Sale Price Predictor

A modern Streamlit application for predicting Barcelona property prices.
Powered by a cloud-deployed ML API.
"""
import streamlit as st
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Page config must be first Streamlit command
st.set_page_config(
    page_title="Barcelona Sale Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Import components
from app.components.ui import render_header, render_result_card
from app.components.forms import render_input_form
from app.services.api_client import get_api_client, check_api_health


def main():
    """Main application entry point."""
    
    # Render header
    render_header()
    st.markdown("---")
    
    # Check API health (cached check)
    api_healthy = check_api_health()
    
    # Layout
    col_left, col_right = st.columns([1, 1.5], gap="large")
    
    with col_left:
        # Input Form
        features = render_input_form()
    
    with col_right:
        # Results Area
        st.markdown("### Market Intelligence")
        
        if features is None:
            # Welcome state
            st.info("👈 Fill out the property details on the left to get an instant valuation.")
            
            st.markdown("""
            <div style="background-color: #f8fafc; padding: 20px; border-radius: 10px; 
                        border: 1px solid #e2e8f0; margin-top: 20px;">
                <h4 style="margin-top:0;">Why trust Lil Homey?</h4>
                <p style="font-size: 0.95rem; color: #475569;">
                    We don't just guess numbers. We analyze <b>neighborhood wealth indices</b> 
                    and <b>population density</b> alongside thousands of real listings to give 
                    you a valuation that reflects the <i>true vibe</i> of the area.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # API Status indicator
            with st.expander("🔌 API Status"):
                if api_healthy:
                    st.success("✅ Connected to prediction API")
                    client = get_api_client()
                    model_info = client.get_model_info()
                    if model_info:
                        st.json({
                            "model_type": model_info.get("model_type"),
                            "version": model_info.get("version"),
                            "features": len(model_info.get("features", [])),
                        })
                else:
                    st.warning("⚠️ API is warming up... First request may take ~30 seconds.")
        else:
            # Prediction state
            if not api_healthy:
                st.warning("⏳ API is starting up (free tier cold start). Please wait...")
            
            # Call the API
            with st.spinner("🔮 Consulting the oracle..."):
                client = get_api_client()
                response = client.predict(features)
            
            if response and response.is_valid:
                # Display result using existing UI component
                render_result_card(
                    prediction=response.predicted_price,
                    range_pct=0.10  # Will be overridden by actual confidence interval
                )
                
                # Show confidence interval from model
                st.caption(
                    f"*Model confidence: €{response.confidence_low:,.0f} - €{response.confidence_high:,.0f}*"
                )
                
                # Explanation section
                st.markdown("#### What drove this price?")
                st.markdown(f"""
                This valuation is influenced by the **{features['neighborhood']}** location profile:
                - **Wealth Factor:** {features['avg_income_index']:.2f} (Index)
                - **Density Factor:** {features['density_val']:.2f} (Pop/km²)
                - **Property Size:** {features['size']} m²
                
                *A higher wealth factor typically drives prices up, while extreme density 
                can have variable effects.*
                """)
                
                # Model version badge
                st.markdown(
                    f"<div style='text-align: right; color: #94a3b8; font-size: 0.75rem;'>"
                    f"Model v{response.model_version}</div>",
                    unsafe_allow_html=True
                )
            else:
                st.error("Could not generate prediction. Please check your inputs and try again.")


if __name__ == "__main__":
    main()
