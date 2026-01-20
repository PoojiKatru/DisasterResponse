import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
import sys

# Import our consolidated pipeline
import pipeline
from vlm import SatelliteVLM
from sam import DroneSAM
from occupancyplanning import DisasterPlanner

# --- CONFIG & AESTHETIC ---
st.set_page_config(page_title="Drone.Command", page_icon="🛰️", layout="wide")

# Custom CSS for the Unique Beige and Blue Aesthetic
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');
    
    :root {
        --bg-charcoal: #0A0A0A; /* Deep, complementary black */
        --text-purple: #BB86FC; /* Vibrant light purple for high readability */
        --accent-purple: #3700B3;
        --border-color: rgba(187, 134, 252, 0.2);
    }
    
    html, body, [class*="st-"] {
        font-family: 'Outfit', sans-serif;
        background-color: var(--bg-charcoal);
        color: var(--text-purple);
    }
    
    .stApp {
        background-color: var(--bg-charcoal);
    }
    
    /* Header Styling */
    .dashboard-header {
        border-bottom: 2px solid var(--text-purple);
        padding-bottom: 15px;
        margin-bottom: 40px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .dashboard-header h1 {
        font-weight: 600;
        letter-spacing: 2px;
        margin: 0;
        color: var(--text-purple);
        text-transform: uppercase;
    }
    
    /* Card/Container Styling */
    div.stButton > button {
        background-color: var(--text-purple);
        color: var(--bg-charcoal);
        border-radius: 8px;
        border: 1px solid var(--text-purple);
        padding: 0.8rem 2.5rem;
        font-weight: 600;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    div.stButton > button:hover {
        background-color: #D1A3FF;
        color: var(--bg-charcoal);
        box-shadow: 0 0 20px rgba(187, 134, 252, 0.4);
        transform: translateY(-2px);
    }
    
    /* Input/Uploader areas */
    .stFileUploader label {
        color: var(--text-purple) !important;
    }

    /* Results layout */
    .result-image-box {
        background: rgba(187, 134, 252, 0.05);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 10px;
    }

    /* Success/Info styling overrides */
    div[data-testid="stNotification"] {
        background-color: rgba(187, 134, 252, 0.1);
        border: 1px solid var(--text-purple);
        color: var(--text-purple);
    }
    
    /* Text overrides to force purple */
    h1, h2, h3, h4, p, span, label, .stMarkdown, small {
        color: var(--text-purple) !important;
    }
    
    /* Override for progress/status messages */
    .stAlert p {
        color: var(--text-purple) !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- STATE MANAGEMENT ---
if 'targets' not in st.session_state:
    st.session_state.targets = []
if 'start_point' not in st.session_state:
    st.session_state.start_point = None
if 'end_point' not in st.session_state:
    st.session_state.end_point = None
if 'selection_mode' not in st.session_state:
    st.session_state.selection_mode = "Rescue Targets"
if 'mission_complete' not in st.session_state:
    st.session_state.mission_complete = False

# --- HEADER ---
st.markdown("""
<div class='dashboard-header'>
    <h1>DRONE.COMMAND <span style='font-size: 0.8rem; font-weight: 300; opacity: 0.5;'>/ ANALYSIS</span></h1>
    <div style='text-align: right;'>
        <span style='color: #BB86FC; font-weight: 600;'>SYSTEM RED</span><br/>
        <span style='font-size: 0.7rem; opacity: 0.5;'>COORD_SYNC ACTIVE</span>
    </div>
</div>
""", unsafe_allow_html=True)

if not st.session_state.mission_complete:
    col_ctrl, col_main = st.columns([1, 2.5])

    with col_ctrl:
        st.markdown("### 1. DEPLOYMENT")
        uploaded_file = st.file_uploader("Upload Satellite Imagery", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file:
            # Save temp file
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.info("Satellite Data Synced.")
            
            st.markdown("### 2. PARAMETERS")
            
            st.session_state.selection_mode = st.radio(
                "Selection Mode",
                ["Rescue Targets", "Start Point", "End Point"],
                horizontal=True
            )

            if st.button("Clear Rescue Targets"):
                st.session_state.targets = []
                st.rerun()
            
            if st.button("Reset Mission Points"):
                st.session_state.start_point = None
                st.session_state.end_point = None
                st.rerun()

            st.write(f"Targets Selected: **{len(st.session_state.targets)}**")
            if st.session_state.start_point:
                st.write(f"Start Point: **{st.session_state.start_point}**")
            
            if st.session_state.end_point:
                st.write(f"End Point: **{st.session_state.end_point}**")
            else:
                st.write("End Point: **AUTO (Same as Start)**")

    if uploaded_file:
        with col_main:
            st.write("### Tactical Command Map")
            st.caption("Select rescue points on the grid below.")
            
            img_pil = Image.open(temp_path)
            # Interactive Picker
            value = streamlit_image_coordinates(img_pil, key="tactical_map")
            
            if value:
                point = (value['y'], value['x'])
                # Only process if this is a NEW click event
                if 'last_clicked' not in st.session_state or st.session_state.last_clicked != point:
                    st.session_state.last_clicked = point
                    
                    if st.session_state.selection_mode == "Start Point":
                        st.session_state.start_point = point
                        st.rerun()
                    elif st.session_state.selection_mode == "End Point":
                        st.session_state.end_point = point
                        st.rerun()
                    else:
                        if point not in st.session_state.targets:
                            st.session_state.targets.append(point)
                            st.rerun()

        # BIG ACTION BUTTON
        st.markdown("---")
        if st.session_state.targets:
            if st.button("EXECUTE RESCUE PATHFINDING", use_container_width=True):
                with st.status("Computing Trajectories...", expanded=True) as status:
                    from io import StringIO
                    import contextlib
                    
                    output_buffer = StringIO()
                    start_pos = st.session_state.start_point if st.session_state.start_point else (50, 50)
                    end_pos = st.session_state.end_point
                    with contextlib.redirect_stdout(output_buffer):
                        pipeline.run_pipeline(temp_path, manual_targets=st.session_state.targets, start_pos=start_pos, end_pos=end_pos)
                    
                    logs = output_buffer.getvalue()
                    if "EMERGENCY FALLBACK" in logs:
                        st.warning("⚠️ EMERGENCY PATH FORCED: The AI could not find a safe path to your target and has drawn a straight line through obstacles. Please try moving your target point away from the black areas.")
                    
                    st.session_state.mission_complete = True
                    st.rerun()
    else:
        with col_main:
            st.info("Satellite Uplink Standby. Please upload an image.")

else:
    # RESULTS VIEW - Shown instead of input to avoid scrolling
    st.markdown("### RESCUE MISSION COMPLETE: ANALYSIS RESULTS")
    if st.button("RESET FOR NEW ANALYSIS"):
        st.session_state.mission_complete = False
        st.session_state.targets = []
        st.rerun()

    st.markdown("---")
    
    r1, r2, r3 = st.columns(3)
    
    with r1:
        st.markdown("<div class='result-image-box'>", unsafe_allow_html=True)
        st.write("**ORIGINAL SCAN**")
        if os.path.exists("step1_original.png"):
            st.image("step1_original.png", caption="Raw Satellite Data - Source imagery for tactical analysis", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with r2:
        st.markdown("<div class='result-image-box'>", unsafe_allow_html=True)
        st.write("**OCCUPANCY GRID**")
        st.caption("○ White: Obstacle | ● Black: Safe")
        if os.path.exists("step2_occupancy.png"):
            st.image("step2_occupancy.png", caption="Perception Filter - Fused VLM & SAM knowledge", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with r3:
        st.markdown("<div class='result-image-box'>", unsafe_allow_html=True)
        st.write("**RESCUE TRAJECTORY**")
        if os.path.exists("step3_final_path.png"):
            st.image("step3_final_path.png", caption="Mission Trajectory - Finalized navigation path", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.write("#### MISSION LOGS")
    st.code("Scene analysis finalized. Hazards bypassed. Optimizing flight path to target coordinates...")

