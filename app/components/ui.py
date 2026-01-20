import streamlit as st
from app.utils.formatting import format_euro

def render_header():
    """Renders the application header."""
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6B7280;
        margin-bottom: 2rem;
    }
    </style>
    <div class="main-header">Barcelona Property Valuator</div>
    <div class="sub-header">Professional AI-powered valuation tool for real estate matching market standards.</div>
    """, unsafe_allow_html=True)

def render_result_card(prediction, range_pct=0.10):
    """
    Renders the valuation result card.
    
    Args:
        prediction (float): The predicted price.
        range_pct (float): The uncertainty range (default ±10%).
    """
    if prediction <= 0:
        st.warning("Could not generate a valid prediction. Please check inputs.")
        return

    lower_bound = prediction * (1 - range_pct)
    upper_bound = prediction * (1 + range_pct)
    
    st.markdown("---")
    st.subheader(" Valuation Result")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(f"""
        <div style="background-color: #F0F9FF; border: 1px solid #BAE6FD; border-radius: 10px; padding: 20px; text-align: center;">
            <div style="color: #64748B; font-size: 1rem; font-weight: 500;">Estimated Market Value</div>
            <div style="color: #0369A1; font-size: 2.5rem; font-weight: 800; margin: 10px 0;">{format_euro(prediction)}</div>
            <div style="color: #64748B; font-size: 0.9rem;">Likely Range: <b>{format_euro(lower_bound)} - {format_euro(upper_bound)}</b></div>
        </div>
        """, unsafe_allow_html=True)
    
    st.caption("Estimation based on current market trends, neighborhood income index, and population density.")
