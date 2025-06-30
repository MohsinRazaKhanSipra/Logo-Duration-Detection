import streamlit as st
import pandas as pd
import json
import os
from io import BytesIO
from datetime import datetime

# Custom CSS (same as app.py for consistency)
st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <style>
        body {
            font-family: 'Inter', sans-serif;
            padding: 0;
            margin: 0;
        }
        .header {
            background-color: #1e3a8a;
            color: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        .header img {
            height: 2.5rem;
        }
        .header h1 {
            color: white;
        }
        .stButton>button {
            background-color: #2563eb;
            color: white;
            border-radius: 0.375rem;
            padding: 0.75rem;
            font-weight: 600;
            transition: background-color 0.2s;
        }
        .stButton>button:hover {
            background-color: #1d4ed8;
        }
        .delete-button {
            background-color: #dc2626;
        }
        .delete-button:hover {
            background-color: #b91c1c;
        }
        .card {
            background-color: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
        }
        .metadata-card table {
            width: 100%;
            border-collapse: collapse;
        }
        .metadata-card th, .metadata-card td {
            padding: 0.75rem;
            text-align: left;
            font-size: 0.875rem;
            color: #1f2937;
        }
        .metadata-card th {
            font-weight: 600;
            width: 40%;
        }
        .metadata-card td {
            color: #2563eb;
        }
        .metadata-card tr:not(:last-child) {
            border-bottom: 1px solid #e5e7eb;
        }
        .stDownloadButton>button {
            background-color: #2563eb;
            color: white;
            border-radius: 0.375rem;
            font-weight: 600;
        }
        .stDownloadButton>button:hover {
            background-color: #1d4ed8;
        }
    </style>
""", unsafe_allow_html=True)

def load_records():
    if os.path.exists("analysis_records.json"):
        with open("analysis_records.json", "r") as f:
            return json.load(f)
    return []

def save_records(records):
    with open("analysis_records.json", "w") as f:
        json.dump(records, f, indent=2)

def main():
    # Header
    st.markdown("""
<div class="header">
    <img src="https://image.made-in-china.com/202f0j00fditTurWBRzG/Night-Vision-300m-Smart-Auto-Tracking-and-Analysis-IP-PTZ-Camera.webp" alt="Logo">
    <div>
        <h1 class="text-2xl font-bold">Analysis Records</h1>
        <p class="text-sm text-gray-100">View and manage past logo detection analyses</p>
    </div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="text-lg font-semibold text-gray-800">Past Analysis Records</h2>', unsafe_allow_html=True)

    records = load_records()
    if not records:
        st.info("No analysis records found.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # Display records as a selectable list
    record_options = [f"{r['timestamp']} - {r['video_source']}" for r in records]
    selected_record = st.selectbox("Select a record to view details", record_options)

    if selected_record:
        record_index = record_options.index(selected_record)
        record = records[record_index]

        # Display metadata
        st.markdown('<h3 class="text-md font-semibold text-gray-800">Video Metadata</h3>', unsafe_allow_html=True)
        metadata = record.get('metadata', {})
        if metadata:
            st.markdown(
                f"""
                <div class="card metadata-card">
                    <table>
                        <tr><th>Duration</th><td>{metadata.get('Duration', 'N/A')}</td></tr>
                        <tr><th>Resolution</th><td>{metadata.get('Resolution', 'N/A')}</td></tr>
                        <tr><th>FPS</th><td>{metadata.get('FPS', 'N/A')}</td></tr>
                        <tr><th>Total Frames</th><td>{metadata.get('Total Frames', 'N/A')}</td></tr>
                    </table>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown('<div class="card metadata-card"><p class="text-center text-gray-600">No metadata available</p></div>', unsafe_allow_html=True)

        # Display logo analysis
        st.markdown('<h3 class="text-md font-semibold text-gray-800">Logo Analysis</h3>', unsafe_allow_html=True)
        logo_stats = record.get('logo_stats', {})
        if logo_stats:
            rows = []
            for logo, data in logo_stats.items():
                for start, end in data.get('appearances', []):
                    rows.append({
                        "Logo": logo,
                        "Start Time (s)": start,
                        "End Time (s)": end,
                        "Duration (s)": round(end - start, 2),
                        "Total Duration (s)": round(data.get('duration', 0), 2),
                        "Frequency": data.get('frequency', 0)
                    })
            df = pd.DataFrame(rows)
            st.dataframe(df)

            # Download Excel
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Logo Analysis')
            output.seek(0)
            st.download_button(
                label="📊 Download Logo Analysis (Excel)",
                data=output,
                file_name=f"logo_analysis_{record['timestamp'].replace(' ', '_').replace(':', '-')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )
        else:
            st.warning("No logo analysis data available.")

        # Download processed video
        processed_video_path = record.get('processed_video_path')
        if processed_video_path and os.path.exists(processed_video_path):
            with open(processed_video_path, "rb") as f:
                st.download_button(
                    label="📥 Download Processed Video",
                    data=f,
                    file_name=os.path.basename(processed_video_path),
                    mime="video/mp4",
                    type="primary"
                )
        else:
            st.info("No processed video available for this record.")

        # Delete record
        if st.button("🗑️ Delete Record", key="delete_record", help="Permanently delete this analysis record"):
            if processed_video_path and os.path.exists(processed_video_path):
                try:
                    os.remove(processed_video_path)
                except Exception as e:
                    st.error(f"Error deleting video file: {str(e)}")
            records.pop(record_index)
            save_records(records)
            st.success("Record deleted successfully!")
            st.experimental_rerun()

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()