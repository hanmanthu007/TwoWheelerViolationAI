import streamlit as st
import pandas as pd
from ui_styles import render_kpi
import database


def render_overview_page():
    """Renders the executive traffic monitoring dashboard overview."""
    st.markdown("### 📊 Operational Overview & System Telemetry")
    st.markdown(
        "<p style='color: #94a3b8; font-size: 13px; margin-top: -10px;'>"
        "Real-time surveillance metrics aggregated from active SQLite violation records (traffic_violations.db)."
        "</p>",
        unsafe_allow_html=True
    )

    # 1. Fetch real KPIs from SQLite database
    kpis = database.get_dashboard_kpis()

    col1, col2, col3 = st.columns(3)
    with col1:
        render_kpi("Total Vehicles Detected", f"{kpis['total_vehicles_detected']:,}", "Surveillance volume", "blue", "🛵")
    with col2:
        render_kpi("Total Violations Logged", f"{kpis['total_violations']:,}", "Confirmed violation events", "rose", "🚨")
    with col3:
        render_kpi("Total Penalty Levied", f"₹ {kpis['total_fine_amount']:,}", "Motor Vehicles Act 2019", "emerald", "💰")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    col4, col5, col6 = st.columns(3)
    with col4:
        render_kpi("Helmet Violations", f"{kpis['helmet_violations']:,}", "Sec 129 / 194D MV Act", "amber", "🪖")
    with col5:
        render_kpi("Triple-Riding Offenses", f"{kpis['triple_riding_violations']:,}", "Sec 128 / 194C MV Act", "rose", "👥")
    with col6:
        render_kpi("Verified Number Plates", f"{kpis['verified_number_plates']:,}", "EasyOCR + Position Regex", "violet", "🔢")

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # 2. System Engine Health / Architecture Card
    st.markdown("#### ⚡ Surveillance Pipeline Status")
    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    with sc1:
        st.markdown("""
        <div style="background: #111827; border: 1px solid #1e293b; border-radius: 8px; padding: 12px; text-align: center;">
            <div style="font-size: 11px; color: #94a3b8; font-weight: 600;">VEHICLE TRACKING</div>
            <div style="font-size: 14px; font-weight: 700; color: #38bdf8; margin: 4px 0;">YOLOv8 + ByteTrack</div>
            <span class="badge badge-ok">ONLINE</span>
        </div>
        """, unsafe_allow_html=True)
    with sc2:
        st.markdown("""
        <div style="background: #111827; border: 1px solid #1e293b; border-radius: 8px; padding: 12px; text-align: center;">
            <div style="font-size: 11px; color: #94a3b8; font-weight: 600;">HELMET CLASSIFIER</div>
            <div style="font-size: 14px; font-weight: 700; color: #38bdf8; margin: 4px 0;">Custom YOLOv8n</div>
            <span class="badge badge-ok">LOADED</span>
        </div>
        """, unsafe_allow_html=True)
    with sc3:
        st.markdown("""
        <div style="background: #111827; border: 1px solid #1e293b; border-radius: 8px; padding: 12px; text-align: center;">
            <div style="font-size: 11px; color: #94a3b8; font-weight: 600;">HEAD SPATIAL MAPPER</div>
            <div style="font-size: 14px; font-weight: 700; color: #38bdf8; margin: 4px 0;">Upper 30% Head Zone</div>
            <span class="badge badge-ok">ACTIVE</span>
        </div>
        """, unsafe_allow_html=True)
    with sc4:
        st.markdown("""
        <div style="background: #111827; border: 1px solid #1e293b; border-radius: 8px; padding: 12px; text-align: center;">
            <div style="font-size: 11px; color: #94a3b8; font-weight: 600;">PLATE OCR ENGINE</div>
            <div style="font-size: 14px; font-weight: 700; color: #38bdf8; margin: 4px 0;">EasyOCR + Regex P1</div>
            <span class="badge badge-ok">VERIFIED</span>
        </div>
        """, unsafe_allow_html=True)
    with sc5:
        st.markdown("""
        <div style="background: #111827; border: 1px solid #1e293b; border-radius: 8px; padding: 12px; text-align: center;">
            <div style="font-size: 11px; color: #94a3b8; font-weight: 600;">MULTIMODAL AI</div>
            <div style="font-size: 14px; font-weight: 700; color: #38bdf8; margin: 4px 0;">Gemini 2.5 Flash</div>
            <span class="badge badge-info">FALLBACK READY</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # 3. Recent Violations Live Feed
    st.markdown("#### 🚨 Recent Recorded Violations")
    challans = database.get_all_challans()

    if not challans:
        st.info("No violation records currently stored in the database. Run an image or video detection to generate records.")
    else:
        recent = challans[:8]
        table_rows = []
        for r in recent:
            table_rows.append({
                "Challan ID": r["challan_id"],
                "Vehicle Plate": r["vehicle_plate"],
                "Date & Time": r["date_time"],
                "Riders": r["rider_count"],
                "Unhelmeted": r["unhelmeted_count"],
                "Fine (INR)": f"₹ {r['total_fine']:,}",
                "Status": r["audit_status"],
                "Source": r["source_type"]
            })

        df = pd.DataFrame(table_rows)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(f"Displaying {len(table_rows)} of {len(challans)} total recorded violations in database.")
