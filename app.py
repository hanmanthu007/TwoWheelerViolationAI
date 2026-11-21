import streamlit as st
import os
import sys

# Ensure current directory is in path for module resolution
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import UI styling and page modules
from ui_styles import apply_custom_css, render_header
from dashboard_pages.overview import render_overview_page
from dashboard_pages.image_detection import render_image_page
from dashboard_pages.video_detection import render_video_page
from dashboard_pages.challan_management import render_challan_page
from dashboard_pages.analytics import render_analytics_page
import database

# 1. Page Configuration
st.set_page_config(
    page_title="TwoWheelerViolationAI - Traffic Surveillance Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Inject Custom Theme CSS
apply_custom_css()

# 3. Top Branding Header
render_header(
    title="Two-Wheeler Traffic Violation AI Platform",
    subtitle="AI-Powered Edge Surveillance & e-Challan Enforcement System"
)

# 4. Sidebar Navigation
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 12px 0 16px 0;">
        <div style="font-size: 32px; margin-bottom: 4px;">🛵🚦</div>
        <div style="font-size: 16px; font-weight: 800; color: #38bdf8; letter-spacing: -0.3px;">TwoWheelerViolationAI</div>
        <div style="font-size: 11px; color: #64748b; font-weight: 500;">Traffic Enforcement Cell</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border: 0; border-top: 1px solid #1e293b; margin: 0 0 16px 0;'>", unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "📊 Dashboard Overview",
            "📸 Image Detection",
            "🎥 Video Detection",
            "📋 Challan Management",
            "📈 Analytics & Reports"
        ],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border: 0; border-top: 1px solid #1e293b; margin: 24px 0 16px 0;'>", unsafe_allow_html=True)

    # Sidebar Engine Telemetry
    st.markdown("#### ⚙️ Edge Runtime")
    st.caption("AI Models: `YOLOv8n` + `helmet_model.pt`")
    st.caption("OCR: `EasyOCR` + Post-OCR Regex")
    st.caption("Tracking: `ByteTrack` (persist=True)")
    st.caption("Database: SQLite `traffic_violations.db`")

    if st.button("🔄 Sync Historical Challans", use_container_width=True):
        database.sync_existing_text_challans()
        st.toast("Synchronized historical receipts from challans/", icon="✅")
        st.rerun()

    st.markdown("""
    <div style="margin-top: 30px; padding: 12px; background: #111827; border-radius: 8px; border: 1px solid #1e293b; font-size: 11px; color: #64748b; text-align: center;">
        Indian Motor Vehicles Act 2019<br/>
        Sec 128 (Triple Riding) &bull; Sec 129 (Helmet)
    </div>
    """, unsafe_allow_html=True)

# 5. Route Page Rendering
if "Dashboard Overview" in page:
    render_overview_page()
elif "Image Detection" in page:
    render_image_page()
elif "Video Detection" in page:
    render_video_page()
elif "Challan Management" in page:
    render_challan_page()
elif "Analytics & Reports" in page:
    render_analytics_page()
