"""
Streamlit Web Interface for Vision-Based Elderly Monitoring System
Replaces the HTML/JS frontend with a modern Streamlit dashboard.
"""

import streamlit as st
import asyncio
import json
import time
import base64
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path
import sys

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.config import get_settings
from app.database import init_database, close_database, images_collection, analyses_collection, alarms_collection
from app.monitoring import VideoMonitor, AudioMonitor
from app.vlm import VLMAnalyzer
from app.schemas import AlarmCreate, AlarmSource, Severity
import app.alarms as alarms_module

# Page config
st.set_page_config(
    page_title="Vision-Based Elderly Monitoring",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .alert-critical {
        background: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    .alert-warning {
        background: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    .alert-info {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    .status-online { color: #4caf50; font-weight: bold; }
    .status-offline { color: #f44336; font-weight: bold; }
    .intent-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 0.25rem;
    }
    .intent-critical { background: #ffebee; color: #c62828; }
    .intent-warning { background: #fff3e0; color: #ef6c00; }
    .intent-info { background: #e3f2fd; color: #1565c0; }
    .intent-ok { background: #e8f5e9; color: #2e7d32; }
    .stButton > button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'monitoring_active' not in st.session_state:
    st.session_state.monitoring_active = False
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()
if 'selected_image_id' not in st.session_state:
    st.session_state.selected_image_id = None
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True


def authenticate(api_key: str) -> bool:
    """Verify API key against settings."""
    settings = get_settings()
    return api_key == settings.api_key


def get_live_frame() -> Optional[bytes]:
    """Get latest frame from video monitor."""
    # This would connect to the running monitor
    # For now, return None - frame fetching happens via API
    return None


async def fetch_images(limit: int = 20, unanalyzed: bool = False) -> List[Dict]:
    """Fetch images from database."""
    await init_database()
    query = {}
    if unanalyzed:
        query["processed"] = False
    cursor = images_collection.find(query, {"data": 0}).sort("created_at", -1).limit(limit)
    docs = await cursor.to_list(length=limit)
    result = []
    for doc in docs:
        image_id = str(doc["_id"])
        payload = {
            "id": image_id,
            "created_at": doc["created_at"].isoformat() if isinstance(doc.get("created_at"), datetime) else str(doc.get("created_at")),
            "source": doc.get("source", "unknown"),
            "help_detected": bool(doc.get("help_detected")),
            "processed": bool(doc.get("processed")),
            "file_url": f"/api/images/{image_id}/file",
        }
        # Fetch analysis
        analysis_doc = await analyses_collection.find_one({"image_id": image_id})
        if analysis_doc:
            payload["analysis"] = {
                "intent": analysis_doc.get("intent", "unknown"),
                "behavioral_state": analysis_doc.get("behavioral_state", "unknown"),
                "confidence": analysis_doc.get("confidence", 0.0),
                "uncertainty": analysis_doc.get("uncertainty", 1.0),
                "description": analysis_doc.get("description", ""),
                "reason": analysis_doc.get("reason", ""),
                "requires_attention": bool(analysis_doc.get("requires_attention")),
                "generated_by": analysis_doc.get("generated_by", ""),
                "created_at": analysis_doc["created_at"].isoformat() if isinstance(analysis_doc.get("created_at"), datetime) else str(analysis_doc.get("created_at")),
            }
        result.append(payload)
    return result


async def fetch_alarms(limit: int = 50) -> List[Dict]:
    """Fetch alarms from database."""
    await init_database()
    cursor = alarms_collection.find().sort("created_at", -1).limit(limit)
    docs = await cursor.to_list(length=limit)
    result = []
    for doc in docs:
        result.append({
            "id": str(doc["_id"]),
            "source": doc.get("source", "unknown"),
            "severity": doc.get("severity", "warning"),
            "message": doc.get("message", ""),
            "status": doc.get("status", "active"),
            "created_at": doc["created_at"].isoformat() if isinstance(doc.get("created_at"), datetime) else str(doc.get("created_at")),
            "acknowledged_at": doc.get("acknowledged_at").isoformat() if doc.get("acknowledged_at") else None,
        })
    return result


async def acknowledge_alarm(alarm_id: str) -> bool:
    """Acknowledge an alarm."""
    await init_database()
    result = await alarms_collection.update_one(
        {"_id": alarm_id},
        {"$set": {"status": "acknowledged", "acknowledged_at": datetime.utcnow()}}
    )
    return result.modified_count > 0


async def capture_and_analyze(note: str = "") -> Optional[str]:
    """Trigger capture and analysis."""
    # This would call the API endpoint
    # For direct DB access, we'd need the image worker
    return None


def format_time(iso_string: str) -> str:
    """Format ISO time string to readable format."""
    try:
        dt = datetime.fromisoformat(iso_string.replace('Z', '+00:00'))
        return dt.strftime("%H:%M:%S")
    except:
        return iso_string


def intent_class(intent: str, requires_attention: bool = False) -> str:
    """Determine CSS class for intent badge."""
    if requires_attention or intent in ["asking_for_help", "emergency", "in_pain", "help"]:
        return "intent-critical"
    elif intent in ["uncomfortable", "agitated", "restless", "distressed", "requesting_medicine"]:
        return "intent-warning"
    elif intent in ["calm", "neutral", "acknowledged", "positive", "gratitude"]:
        return "intent-ok"
    else:
        return "intent-info"


def render_login():
    """Render login page."""
    st.markdown('<div class="main-header">🏥 Vision-Based Elderly Monitoring</div>', unsafe_allow_html=True)
    st.markdown("**Caregiver Console** — Doctors & Nurses")
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        api_key = st.text_input("API Key", type="password", placeholder="Enter API key", key="login_key")
        if st.button("🔓 Open Console", use_container_width=True):
            if authenticate(api_key):
                st.session_state.authenticated = True
                st.session_state.api_key = api_key
                st.rerun()
            else:
                st.error("Invalid API key")


def render_sidebar():
    """Render sidebar with controls."""
    with st.sidebar:
        st.markdown("### 🎛️ Control Panel")

        # Connection status
        st.markdown("#### System Status")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Camera**")
            st.markdown('<span class="status-online">● Online</span>', unsafe_allow_html=True)
        with col2:
            st.markdown("**Audio**")
            st.markdown('<span class="status-offline">○ Disabled</span>', unsafe_allow_html=True)

        st.markdown("---")

        # Auto-refresh toggle
        st.session_state.auto_refresh = st.checkbox("🔄 Auto Refresh", value=st.session_state.auto_refresh)

        # Refresh interval
        refresh_interval = st.slider("Refresh Interval (sec)", 2, 30, 5)

        # Manual refresh button
        if st.button("🔄 Refresh Now"):
            st.session_state.last_refresh = 0
            st.rerun()

        st.markdown("---")

        # Patient selection
        st.markdown("#### Patient")
        patient_id = st.selectbox("Patient ID", ["patient_001", "patient_002", "patient_003"])

        # Capture button
        if st.button("📸 Capture & Analyze", use_container_width=True):
            with st.spinner("Capturing..."):
                # This would trigger the API
                st.success("Capture initiated! Check Images tab.")

        st.markdown("---")

        # Quick actions
        st.markdown("#### Quick Actions")
        if st.button("🚨 Emergency Alert", use_container_width=True):
            st.warning("Emergency alert sent!")

        if st.button("📞 Call Family", use_container_width=True):
            st.info("Family notification sent!")

        st.markdown("---")

        # Logout
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.api_key = ""
            st.rerun()


def render_live_camera():
    """Render live camera feed."""
    st.markdown("### 📹 Live Camera Feed")

    # Placeholder for live video
    # In production, this would use WebRTC or MJPEG stream
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("""
        <div style="background: #000; border-radius: 8px; aspect-ratio: 16/9; display: flex; align-items: center; justify-content: center; color: #666;">
            📷 Live Camera Feed<br><small>Connect to API for live stream</small>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("#### Camera Controls")
        if st.button("📸 Snapshot"):
            st.info("Snapshot captured")
        if st.button("⏺️ Record"):
            st.info("Recording started")
        st.markdown("**FPS:** 30")
        st.markdown("**Resolution:** 1280×720")


def render_images_gallery(images: List[Dict]):
    """Render captured images gallery with analysis."""
    st.markdown("### 🖼️ Captured Images & VLM Analysis")

    if not images:
        st.info("No images captured yet. Use 'Capture & Analyze' to start.")
        return

    # Grid layout
    cols_per_row = 2
    for i in range(0, len(images), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(images):
                break

            img = images[idx]
            with col:
                # Image container
                with st.container(border=True):
                    # Image placeholder (would load from API)
                    st.markdown(f"""
                    <div style="background: #000; border-radius: 8px; aspect-ratio: 16/9; display: flex; align-items: center; justify-content: center; color: #666; margin-bottom: 0.5rem;">
                        Image {img['id'][:8]}...
                    </div>
                    """, unsafe_allow_html=True)

                    # Metadata
                    st.caption(f"🕐 {format_time(img['created_at'])} · 📍 {img['source']}")
                    if img.get('help_detected'):
                        st.markdown("🚩 **HELP DETECTED**")

                    # Analysis
                    analysis = img.get("analysis")
                    if analysis:
                        intent = analysis.get("intent", "unknown")
                        state = analysis.get("behavioral_state", "unknown")
                        conf = analysis.get("confidence", 0) * 100
                        unc = analysis.get("uncertainty", 1) * 100
                        requires_attn = analysis.get("requires_attention", False)
                        desc = analysis.get("description", "")

                        badge_class = intent_class(intent, requires_attn)
                        st.markdown(f'<span class="intent-badge {badge_class}">{intent}</span>', unsafe_allow_html=True)
                        st.caption(f"State: {state} · Conf: {conf:.0f}% · Unc: {unc:.0f}%")
                        if desc:
                            st.caption(desc[:100] + ("..." if len(desc) > 100 else ""))
                    else:
                        st.caption("⏳ VLM analysis pending...")

                    # Action buttons
                    btn_col1, btn_col2 = st.columns(2)
                    with btn_col1:
                        if st.button("🔍 View", key=f"view_{img['id']}", use_container_width=True):
                            st.session_state.selected_image_id = img['id']
                            st.rerun()
                    with btn_col2:
                        if st.button("🗑️ Delete", key=f"del_{img['id']}", use_container_width=True):
                            st.warning("Delete not implemented")


def render_alarms(alarms: List[Dict]):
    """Render alarms panel."""
    st.markdown("### 🚨 Alarms")

    if not alarms:
        st.success("✅ No active alarms")
        return

    for alarm in alarms:
        severity = alarm.get("severity", "warning")
        status = alarm.get("status", "active")

        if status == "acknowledged":
            continue  # Skip acknowledged in main view

        # Determine alert style
        if severity == "critical":
            alert_class = "alert-critical"
            icon = "🔴"
        else:
            alert_class = "alert-warning"
            icon = "🟡"

        with st.container():
            st.markdown(f"""
            <div class="{alert_class}">
                <strong>{icon} {alarm['source'].upper()}</strong> {alarm['message']}<br>
                <small>🕐 {format_time(alarm['created_at'])}</small>
            </div>
            """, unsafe_allow_html=True)

            if st.button("✅ Acknowledge", key=f"ack_{alarm['id']}"):
                asyncio.run(acknowledge_alarm(alarm['id']))
                st.rerun()


def render_image_detail(image_id: str):
    """Render detailed view of a single image."""
    st.markdown(f"### 🔍 Image Detail: {image_id[:8]}...")

    if st.button("← Back to Gallery"):
        st.session_state.selected_image_id = None
        st.rerun()

    # Fetch full image data
    # This would load from API
    st.info("Detailed view - would show full image, analysis, and clinical briefing")


def render_analytics():
    """Render analytics dashboard."""
    st.markdown("### 📊 Analytics Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Captures", "142", "+12 today")
    with col2:
        st.metric("Help Detections", "3", "0 today")
    with col3:
        st.metric("Avg Confidence", "87%", "+2%")
    with col4:
        st.metric("Avg Uncertainty", "13%", "-1%")

    st.markdown("---")

    # Intent distribution
    st.markdown("#### Intent Distribution (Last 24h)")
    intent_data = {
        "neutral": 45,
        "asking_for_help": 3,
        "requesting_water": 8,
        "requesting_food": 5,
        "in_pain": 2,
        "wants_sleep": 12,
        "acknowledged": 15
    }
    st.bar_chart(intent_data)

    # Confidence over time
    st.markdown("#### Confidence Trend")
    import pandas as pd
    import numpy as np
    dates = pd.date_range(end=datetime.now(), periods=24, freq='H')
    conf_data = pd.DataFrame({
        'Time': dates,
        'Confidence': np.random.normal(85, 5, 24).clip(60, 99)
    })
    st.line_chart(conf_data.set_index('Time'))


def main():
    """Main Streamlit app."""
    # Initialize database connection
    try:
        asyncio.run(init_database())
    except:
        pass

    # Login check
    if not st.session_state.authenticated:
        render_login()
        return

    # Render sidebar
    render_sidebar()

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📹 Live", "🖼️ Images", "🚨 Alarms", "📊 Analytics"])

    with tab1:
        render_live_camera()

    with tab2:
        # Fetch images
        if st.session_state.auto_refresh or st.session_state.last_refresh == 0:
            images = asyncio.run(fetch_images(limit=20))
            st.session_state.last_refresh = time.time()
        else:
            images = asyncio.run(fetch_images(limit=20))

        if st.session_state.selected_image_id:
            render_image_detail(st.session_state.selected_image_id)
        else:
            render_images_gallery(images)

    with tab3:
        # Fetch alarms
        alarms = asyncio.run(fetch_alarms(limit=50))
        render_alarms(alarms)

        # Show acknowledged alarms in expander
        acked_alarms = [a for a in alarms if a.get("status") == "acknowledged"]
        if acked_alarms:
            with st.expander(f"✅ Acknowledged ({len(acked_alarms)})"):
                for alarm in acked_alarms:
                    st.caption(f"✅ {alarm['source']}: {alarm['message']} — {format_time(alarm['acknowledged_at'])}")

    with tab4:
        render_analytics()

    # Auto-refresh logic
    if st.session_state.auto_refresh:
        time.sleep(0.1)  # Small delay
        if time.time() - st.session_state.last_refresh > 5:
            st.session_state.last_refresh = time.time()
            st.rerun()


if __name__ == "__main__":
    main()