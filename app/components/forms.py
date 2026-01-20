import streamlit as st
from app.services.data_service import get_districts, get_neighborhoods, get_neighborhood_id, get_socio_metrics

PROPERTY_TYPES = ['flat', 'penthouse', 'duplex', 'studio', 'chalet', 'countryHouse']

def render_input_form():
    """
    Renders the property input form sidebar/main section.
    Returns:
        dict: A dictionary of features if form is submitted, else None.
    """
    st.subheader("Property Details")
    
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            size = st.number_input("Size (m²)", min_value=15, max_value=1000, value=80, step=1, help="Total built area in square meters.")
            rooms = st.number_input("Rooms", min_value=0, max_value=20, value=3, step=1)
            floor = st.number_input("Floor Level", min_value=0, max_value=50, value=1, step=1, help="0 for ground floor.")
            
        with col2:
            bathrooms = st.number_input("Bathrooms", min_value=0, max_value=10, value=1, step=1)
            prop_type = st.selectbox("Property Type", options=PROPERTY_TYPES, index=0)
            
            # Extra fields not in model but requested
            has_lift = st.checkbox("Elevator", value=True)
            has_parking = st.checkbox("Parking", value=False)
            
    st.subheader("Location")
    
    # Cascading dropdowns
    districts = get_districts()
    if not districts:
        st.error("No location data found. Please check data files.")
        return None
        
    district = st.selectbox("District", options=districts)
    
    neighborhoods = get_neighborhoods(district)
    neighborhood = st.selectbox("Neighborhood", options=neighborhoods)
    
    # Get hidden features
    neighborhood_id = get_neighborhood_id(neighborhood)
    socio_metrics = get_socio_metrics(neighborhood_id)
    
    # Hidden info for user transparency (optional, maybe in expander)
    with st.expander("Location Insights (Auto-filled)"):
        st.write(f"**District:** {district}")
        st.write(f"**Neighborhood ID:** {neighborhood_id}")
        st.metric("Avg. Income Index", f"{socio_metrics.get('income', 0):.2f}")
        st.metric("Population Density", f"{socio_metrics.get('density', 0):.2f}")
        
    submit = st.button("Estimate Price", type="primary", use_container_width=True)
    
    if submit:
        return {
            "size": size,
            "rooms": rooms,
            "bathrooms": bathrooms,
            "neighborhood": neighborhood,
            "propertyType": prop_type,
            "district": district,
            "avg_income_index": socio_metrics.get('income', 0),
            "density_val": socio_metrics.get('density', 0),
            # Extras for display/logging
            "has_lift": has_lift, 
            "has_parking": has_parking,
            "floor": floor
        }
    return None
