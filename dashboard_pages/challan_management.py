import streamlit as st
import pandas as pd
import os
import json
from datetime import datetime

from pdf_challan_generator import generate_pdf_challan
import database


def render_challan_page():
    """Renders the e-Challan Management, Search, and PDF Export page."""
    st.markdown("### 📋 E-Challan Registry & Statutory Notice Management")
    st.markdown(
        "<p style='color: #94a3b8; font-size: 13px; margin-top: -10px;'>"
        "Search, filter, inspect photographic evidence, and download court-ready PDF e-Challans generated under Section 136A of the Motor Vehicles Act."
        "</p>",
        unsafe_allow_html=True
    )

    # 1. Search & Filter Bar
    f_col1, f_col2, f_col3 = st.columns([1.5, 1, 1])
    with f_col1:
        search_query = st.text_input("🔍 Search by Registration / Vehicle Plate:", placeholder="e.g. TS09, DL, AP01...")
    with f_col2:
        viol_filter = st.selectbox(
            "Filter by Violation Type:",
            ["All", "Helmet Violation", "Triple Riding", "Both Violations"]
        )
    with f_col3:
        status_filter = st.selectbox(
            "Filter by Verification Status:",
            ["All", "CONFIRMED / VERIFIED", "REVIEW_REQUIRED / PENDING", "HISTORICAL"]
        )

    # Query database
    all_challans = database.get_all_challans(
        search_plate=search_query if search_query else None,
        violation_filter=viol_filter
    )

    # Apply status filter in memory if selected
    if status_filter != "All":
        if "CONFIRMED" in status_filter or "VERIFIED" in status_filter:
            all_challans = [c for c in all_challans if c["audit_status"] in ["CONFIRMED", "VERIFIED", "LOCAL_OCR_VALIDATED", "LOCAL_OCR_VALID"]]
        elif "REVIEW" in status_filter or "PENDING" in status_filter:
            all_challans = [c for c in all_challans if "REVIEW" in c["audit_status"] or "PENDING" in c["audit_status"]]
        elif "HISTORICAL" in status_filter:
            all_challans = [c for c in all_challans if c["source_type"] == "HISTORICAL"]

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    if not all_challans:
        st.info("No challans match the specified search or filter criteria.")
        return

    # 2. Challan Records Table
    st.markdown(f"#### 📄 Registered Violations ({len(all_challans)} Found)")

    table_data = []
    for c in all_challans:
        table_data.append({
            "Challan ID": c["challan_id"],
            "Vehicle Plate": c["vehicle_plate"],
            "Date & Time": c["date_time"],
            "Riders": c["rider_count"],
            "No Helmet": c["unhelmeted_count"],
            "Triple Riding": "Yes" if c["triple_riding"] else "No",
            "Fine (INR)": f"₹ {c['total_fine']:,}",
            "Status": c["audit_status"],
            "Source": c["source_type"]
        })

    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Export to CSV button
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Registry to CSV",
        data=csv_bytes,
        file_name=f"challans_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # 3. Detailed Challan Inspector & PDF Exporter
    st.markdown("#### 🔍 Detailed Record Inspector & PDF Notice Issuance")

    challan_options = [c["challan_id"] for c in all_challans]
    selected_id = st.selectbox("Select Challan ID to Inspect:", challan_options)

    selected_rec = next((c for c in all_challans if c["challan_id"] == selected_id), None)

    if selected_rec:
        insp_col1, insp_col2 = st.columns([1.1, 0.9])

        with insp_col1:
            st.markdown("<b>Surveillance Evidence Snapshot:</b>", unsafe_allow_html=True)
            ev_file = selected_rec.get("evidence_path", "")
            if ev_file and os.path.exists(ev_file):
                st.image(ev_file, caption=f"Evidence for {selected_rec['vehicle_plate']} ({selected_rec['challan_id']})", use_container_width=True)
            elif os.path.exists("final_violation_output.jpg"):
                st.image("final_violation_output.jpg", caption="Surveillance Snapshot", use_container_width=True)
            else:
                st.info("No photographic image file linked to this historical entry.")

        with insp_col2:
            st.markdown("""
            <div class="content-card">
                <div class="card-title">📜 Statutory Violation Notice Summary</div>
            """, unsafe_allow_html=True)

            st.markdown(f"<b>Challan Reference ID:</b> <code>{selected_rec['challan_id']}</code>", unsafe_allow_html=True)
            st.markdown(f"<b>Registration Number:</b> <div class='plate-display' style='font-size: 18px; margin: 4px 0;'>{selected_rec['vehicle_plate']}</div>", unsafe_allow_html=True)
            st.markdown(f"<b>Notice Date & Time:</b> {selected_rec['date_time']}", unsafe_allow_html=True)
            st.markdown(f"<b>Riders Detected:</b> {selected_rec['rider_count']} (Unhelmeted: {selected_rec['unhelmeted_count']})", unsafe_allow_html=True)

            # Violations list
            v_json = selected_rec.get("violations_json", "[]")
            try:
                v_list = json.loads(v_json) if isinstance(v_json, str) else v_json
            except Exception:
                v_list = [v_json]

            st.markdown("<b>Offenses Charged:</b>", unsafe_allow_html=True)
            if v_list:
                for v in v_list:
                    item_str = v[0] if isinstance(v, (list, tuple)) else str(v)
                    st.markdown(f"• {item_str}")
            else:
                st.markdown("• General Motor Vehicle Safety Violation")

            st.markdown(f"<h3 style='color: #f43f5e; margin: 12px 0 8px 0;'>TOTAL FINE: ₹ {selected_rec['total_fine']:,}</h3>", unsafe_allow_html=True)

            st.markdown(f"<b>Verification Status:</b> <span class='badge badge-ok'>{selected_rec['audit_status']}</span>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Action Buttons: Download PDF and Delete
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                # Generate official PDF on the fly
                pdf_bytes = generate_pdf_challan(selected_rec)
                st.download_button(
                    label="📄 Download Official PDF Notice",
                    data=pdf_bytes,
                    file_name=f"Official_Notice_{selected_rec['vehicle_plate']}_{selected_rec['challan_id']}.pdf",
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True
                )

            with btn_col2:
                if st.button("🗑️ Delete Challan Record", use_container_width=True):
                    deleted = database.delete_challan(selected_rec["challan_id"])
                    if deleted:
                        st.toast(f"Record {selected_rec['challan_id']} deleted from database.", icon="🗑️")
                        st.rerun()
                    else:
                        st.error("Could not delete record.")
