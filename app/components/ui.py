import streamlit as st
from app.utils.formatting import format_euro

def render_custom_css():
    """Injects custom CSS for Lil Homey branding."""
    st.markdown("""
    <style>
        /* Import Font: Outfit */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
        }

        /* Branding Colors */
        :root {
            --primary-color: #6366f1; /* Indigo */
            --accent-color: #a855f7; /* Purple */
            --bg-color: #f8fafc;
        }

        /* Main Container Styling */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        /* Header Styling */
        .brand-header {
            background: linear-gradient(135deg, var(--primary-color), var(--accent-color));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem;
            font-weight: 800;
            letter-spacing: -0.05rem;
            margin-bottom: 0.2rem;
        }
        .brand-subheader {
            font-size: 1.1rem;
            color: #64748b;
            font-weight: 400;
            margin-bottom: 2.5rem;
        }

        /* Card Styling */
        .stCard {
            background-color: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            border: 1px solid #e2e8f0;
        }

        /* Metric Styling */
        .big-number {
            font-size: 3.5rem;
            font-weight: 800;
            color: #1e293b;
        }
        .metric-label {
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-size: 0.75rem;
            font-weight: 600;
            color: #94a3b8;
        }
    </style>
    """, unsafe_allow_html=True)

def render_header():
    """Renders the branded application header."""
    render_custom_css()
    st.markdown("""
    <div>
        <div class="brand-header">Lil Homey</div>
        <div class="brand-subheader">Your laid-back expert for Barcelona real estate.</div>
    </div>
    """, unsafe_allow_html=True)

def render_result_card(prediction, range_pct=0.10):
    """
    Renders the valuation result in a sleek, non-AI looking card.
    """
    if prediction <= 0:
        st.warning("Could not generate a valid prediction. Double check your inputs.")
        return

    lower_bound = prediction * (1 - range_pct)
    upper_bound = prediction * (1 + range_pct)
    
    st.markdown("---")
    
    # Modern Result Card
    st.markdown(f"""
    <div style="background: white; border-radius: 16px; padding: 30px; border: 1px solid #e2e8f0; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);">
        <div style="text-align: center;">
            <div style="color: #6366f1; font-weight: 600; letter-spacing: 1px; font-size: 0.9rem; text-transform: uppercase; margin-bottom: 10px;">The Verdict</div>
            <div style="font-size: 3.5rem; font-weight: 800; color: #0f172a; line-height: 1;">{format_euro(prediction)}</div>
            <div style="color: #64748b; margin-top: 10px; font-size: 1.1rem;">
                Estimated Range: <span style="color: #0f172a; font-weight: 600;">{format_euro(lower_bound)} - {format_euro(upper_bound)}</span>
            </div>
            <div style="margin-top: 20px; font-size: 0.9rem; color: #94a3b8; font-style: italic;">
                "Based on recent data in your chosen neighborhood."
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
