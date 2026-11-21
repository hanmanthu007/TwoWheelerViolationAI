import streamlit as st

def apply_custom_css():
    """Injects high-end, responsive dark theme CSS into the Streamlit app."""
    css = """
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main background & container */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
    }

    /* Top Header Bar */
    .traffic-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px 28px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .traffic-title {
        font-size: 26px;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .traffic-subtitle {
        font-size: 13px;
        color: #94a3b8;
        margin-top: 4px;
        font-weight: 500;
    }

    /* KPI Stat Cards */
    .kpi-card {
        background: #131c2e;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
        position: relative;
        overflow: hidden;
    }

    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: #38bdf8;
    }

    .kpi-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
    }

    .kpi-blue::before { background: #38bdf8; }
    .kpi-rose::before { background: #f43f5e; }
    .kpi-amber::before { background: #f59e0b; }
    .kpi-emerald::before { background: #10b981; }
    .kpi-violet::before { background: #a855f7; }

    .kpi-label {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94a3b8;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .kpi-value {
        font-size: 28px;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.5px;
        line-height: 1.1;
    }

    .kpi-subtext {
        font-size: 11px;
        color: #64748b;
        margin-top: 6px;
        font-weight: 500;
    }

    /* Glass Cards */
    .content-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.2);
    }

    .card-title {
        font-size: 16px;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
        border-bottom: 1px solid #1e293b;
        padding-bottom: 10px;
    }

    /* Status Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.3px;
        text-transform: uppercase;
    }

    .badge-ok {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .badge-viol {
        background: rgba(244, 63, 94, 0.15);
        color: #fb7185;
        border: 1px solid rgba(244, 63, 94, 0.3);
    }

    .badge-warn {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    .badge-info {
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }

    /* License Plate Box Display */
    .plate-display {
        font-family: 'JetBrains Mono', monospace;
        font-size: 22px;
        font-weight: 800;
        background: #f8fafc;
        color: #0f172a;
        padding: 8px 18px;
        border-radius: 6px;
        border: 3px solid #0f172a;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        display: inline-block;
        letter-spacing: 2px;
        text-align: center;
    }

    /* Streamlit overrides */
    div[data-testid="stSidebar"] {
        background-color: #0c1220;
        border-right: 1px solid #1e293b;
    }

    div[data-testid="stMetricValue"] {
        font-size: 28px !important;
        font-weight: 800 !important;
        color: #38bdf8 !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0f172a;
        padding: 6px;
        border-radius: 10px;
        border: 1px solid #1e293b;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 600;
        padding: 8px 16px;
    }

    .stTabs [aria-selected="true"] {
        background-color: #1e293b !important;
        color: #38bdf8 !important;
    }

    /* Custom Table Styling */
    .dataframe {
        background-color: #111827 !important;
        color: #f1f5f9 !important;
        border: 1px solid #1f2937 !important;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_header(title="Two-Wheeler Traffic Violation AI Platform", subtitle="Automated Edge Surveillance & e-Challan Enforcement System"):
    """Renders the top branding header."""
    html = f"""
    <div class="traffic-header">
        <div>
            <h1 class="traffic-title">🚦 {title}</h1>
            <div class="traffic-subtitle">{subtitle} &bull; Powered by YOLOv8 &bull; EasyOCR &bull; Gemini Vision &bull; ByteTrack</div>
        </div>
        <div style="text-align: right;">
            <span class="badge badge-ok">SURVEILLANCE ONLINE</span>
            <div style="font-size: 11px; color: #64748b; margin-top: 5px;">M.V. Act 2019 Ready</div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_kpi(label, value, subtext="", color="blue", icon="📊"):
    """Renders a modern stat KPI card."""
    html = f"""
    <div class="kpi-card kpi-{color}">
        <div class="kpi-label">{icon} {label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-subtext">{subtext}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
