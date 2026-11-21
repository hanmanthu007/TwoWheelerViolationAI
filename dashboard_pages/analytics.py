
import streamlit as st
import pandas as pd
import database
from ui_styles import render_kpi


def render_analytics_page():
    """Display enforcement analytics from the SQLite database."""

    st.markdown("### Violation Analytics, Hotspot Trends & Repeat Offenders")
    st.caption(
        "Statistical breakdown of records currently stored in "
        "traffic_violations.db."
    )

    metrics = database.get_analytics_metrics()
    kpis = database.get_dashboard_kpis()

    # 1. KPI SUMMARY
    total_fine = kpis.get("total_fine_amount", 0) or 0
    total_violations = kpis.get("total_violations", 0) or 0
    verified_plates = kpis.get("verified_number_plates", 0) or 0

    avg_fine = total_fine / total_violations if total_violations else 0
    valid_rate = (
        verified_plates / total_violations * 100
        if total_violations else 0
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        render_kpi(
            "Total Assessed Penalties",
            f"₹ {total_fine:,.0f}",
            "Total assessed amount",
            "emerald",
            "₹"
        )

    with col2:
        render_kpi(
            "Average Penalty / Record",
            f"₹ {avg_fine:,.2f}",
            "Average assessed amount",
            "blue",
            "AVG"
        )

    with col3:
        render_kpi(
            "Plate Verification Rate",
            f"{valid_rate:.1f}%",
            "Verified plates / total records",
            "violet",
            "%"
        )

    st.divider()

    # 2. OFFENSE CATEGORY DISTRIBUTION
    st.markdown("#### Offense Category Distribution")

    cat_data = metrics.get("categories", {}) or {}

    cat_df = pd.DataFrame(
        [
            {"Category": str(category), "Records": int(count)}
            for category, count in cat_data.items()
            if count is not None and int(count) > 0
        ]
    )

    if not cat_df.empty:
        cat_df = cat_df.sort_values("Records", ascending=False)
        st.bar_chart(
            cat_df.set_index("Category"),
            color="#f43f5e",
            use_container_width=True
        )
        with st.expander("View offense category data"):
            st.dataframe(cat_df, use_container_width=True, hide_index=True)
    else:
        st.info("No offense category data available.")

    st.divider()

    # 3. SURVEILLANCE SOURCE BREAKDOWN
    st.markdown("#### Surveillance Source Breakdown")

    src_data = metrics.get("sources", {}) or {}

    src_df = pd.DataFrame(
        [
            {"Source": str(source), "Records": int(count)}
            for source, count in src_data.items()
            if count is not None and int(count) > 0
        ]
    )

    if not src_df.empty:
        src_df = src_df.sort_values("Records", ascending=False)
        st.bar_chart(
            src_df.set_index("Source"),
            color="#38bdf8",
            use_container_width=True
        )
        with st.expander("View surveillance source data"):
            st.dataframe(src_df, use_container_width=True, hide_index=True)
    else:
        st.info("No surveillance source data available.")

    st.divider()

    # 4. REPEAT OFFENDERS
    st.markdown("#### Top Repeat Offender Two-Wheelers")
    st.caption(
        "Only verified registration plates with at least two recorded "
        "offenses are included."
    )

    top_offenders = metrics.get("top_offenders", []) or []

    if top_offenders:
        offender_rows = []

        for rank, item in enumerate(top_offenders, start=1):
            count = int(item.get("count", 0) or 0)
            offender_rows.append({
                "Rank": rank,
                "Vehicle Registration": item.get("vehicle_plate", "UNKNOWN"),
                "Violation Count": count,
                "Cumulative Assessed Penalties": (
                    f"₹ {float(item.get('total_fines', 0) or 0):,.0f}"
                ),
                "Priority": "HIGH" if count >= 2 else "MONITORED"
            })

        offender_df = pd.DataFrame(offender_rows)

        st.dataframe(
            offender_df,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info(
            "No verified repeat offenders found. "
            "A vehicle must have at least two verified violation records."
        )

    st.divider()

    # 5. OCR AND VERIFICATION AUDIT
    st.markdown("#### OCR & Verification Audit Quality")

    audit_data = metrics.get("audit_statuses", {}) or {}

    if audit_data:
        audit_items = sorted(audit_data.items())
        columns_per_row = 4

        for start in range(0, len(audit_items), columns_per_row):
            row_items = audit_items[start:start + columns_per_row]
            audit_cols = st.columns(len(row_items))

            for col, (status, count) in zip(audit_cols, row_items):
                with col:
                    st.metric(
                        label=str(status).replace("_", " "),
                        value=int(count)
                    )
    else:
        st.info("No verification audit records available.")
