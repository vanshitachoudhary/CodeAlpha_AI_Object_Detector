import streamlit as st
import numpy as np
import cv2
import time
from PIL import Image
from detector import detect_image, detect_video, get_model

model = get_model()

# ---------------- PAGE ---------------- #

st.set_page_config(
    page_title="AI Vision Object Detector Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- SESSION ---------------- #

if "images" not in st.session_state:
    st.session_state.images = 0

if "videos" not in st.session_state:
    st.session_state.videos = 0

if "objects" not in st.session_state:
    st.session_state.objects = 0

# ---------------- CSS ---------------- #

st.markdown("""

<style>

html,body{

background:#0E1117;

font-family:Arial;

}

.main{

background:#0E1117;

}

.block-container{

max-width:1350px;

padding-top:2rem;

}

section[data-testid="stSidebar"]{

background:#111827;

}

.metric-card{

background:#1F2937;

padding:15px;

border-radius:15px;

text-align:center;

box-shadow:0 0 15px rgba(0,0,0,.35);

}

.hero{

padding:30px;

border-radius:20px;

background:linear-gradient(90deg,#2563EB,#06B6D4);

color:white;

text-align:center;

margin-bottom:20px;

}

footer{

visibility:hidden;

}

</style>

""",unsafe_allow_html=True)

# ---------------- SIDEBAR ---------------- #

with st.sidebar:

    st.title("🤖 AI Vision")

    st.success("Version 4.0")

    st.caption("YOLOv8 + ByteTrack")

    st.divider()

    mode = st.radio(

        "Select Mode",

        [

            "🖼 Image Detection",

            "🎥 Video Detection",

            "📷 Live Webcam"

        ]

    )

    confidence = st.slider(

        "Confidence",

        0.10,

        1.00,

        0.40,

        0.05

    )

    st.divider()

    st.subheader("📊 Dashboard")

    st.metric("Images",st.session_state.images)

    st.metric("Videos",st.session_state.videos)

    st.metric("Objects",st.session_state.objects)

    st.metric("AI Status","🟢 Online")

    st.divider()

    st.write("### AI Engine")

    st.success("YOLOv8 Nano")

    st.info("ByteTrack Enabled")

    st.write("Python 3.11")

# ---------------- HERO ---------------- #

st.markdown("""

<div class="hero">

<h1>

🤖 AI Vision Object Detector Pro

</h1>

<h3>

Deep Learning • Computer Vision • YOLOv8 • ByteTrack

</h3>

<p>

Upload an image, video or use your webcam to detect and track objects using Artificial Intelligence.

</p>

</div>

""",unsafe_allow_html=True)

# ---------------- DASHBOARD ---------------- #

d1,d2,d3,d4=st.columns(4)

with d1:

    st.metric("📷 Images",st.session_state.images)

with d2:

    st.metric("🎥 Videos",st.session_state.videos)

with d3:

    st.metric("📦 Objects",st.session_state.objects)

with d4:

    st.metric("🤖 Model","YOLOv8")
# ---------------- AI FEATURES ---------------- #

st.markdown("## ✨ AI Features")

f1, f2, f3, f4 = st.columns(4)

with f1:
    st.info("""
### 🧠 YOLOv8

Real-Time Object Detection
""")

with f2:
    st.info("""
### 🎯 ByteTrack

Object Tracking IDs
""")

with f3:
    st.info("""
### ⚡ Fast AI

Optimized Inference
""")

with f4:
    st.info("""
### 📊 Analytics

Detection Statistics
""")

st.markdown("---")

# ---------------- IMAGE DETECTION ---------------- #

if mode == "🖼 Image Detection":

    st.subheader("🖼 AI Image Detection")

    uploaded_image = st.file_uploader(

        "Upload Image",

        type=["jpg","jpeg","png"]

    )

    if uploaded_image:

        image = Image.open(uploaded_image).convert("RGB")

        image_np = np.array(image)

        start = time.time()

        with st.spinner("🤖 Detecting Objects..."):

            detected_image, object_counts = detect_image(

                image_np,

                confidence

            )

        end = round(time.time()-start,2)

        total = sum(object_counts.values())

        st.session_state.images += 1

        st.session_state.objects += total

        c1,c2 = st.columns(2)

        with c1:

            st.markdown("### 📷 Original Image")

            st.image(
                image,
                use_container_width=True
            )

        with c2:

            st.markdown("### 🎯 Detection Result")

            st.image(
                detected_image,
                use_container_width=True
            )

        st.markdown("---")

        a1,a2,a3,a4 = st.columns(4)

        with a1:
            st.metric("Objects",total)

        with a2:
            st.metric("Classes",len(object_counts))

        with a3:
            st.metric("Confidence",f"{int(confidence*100)}%")

        with a4:
            st.metric("Time",f"{end}s")

        st.markdown("### 📦 Objects Found")

        if object_counts:

            for obj,count in object_counts.items():

                st.success(f"✅ {obj.title()} : {count}")

        else:

            st.warning("No objects detected.")

        image_bytes = cv2.imencode(
            ".jpg",
            cv2.cvtColor(
                detected_image,
                cv2.COLOR_RGB2BGR
            )
        )[1].tobytes()

        st.download_button(

            "📥 Download Result",

            image_bytes,

            file_name="detected_image.jpg",

            mime="image/jpeg",

            use_container_width=True

        )

st.markdown("---")
# ---------------- VIDEO DETECTION ---------------- #

if mode == "🎥 Video Detection":

    st.subheader("🎥 AI Video Detection & Tracking")

    uploaded_video = st.file_uploader(

        "Upload Video",

        type=["mp4", "avi", "mov"]

    )

    if uploaded_video:

        start = time.time()

        with st.spinner("🤖 Processing video..."):

            output_video, object_counts = detect_video(

                uploaded_video,

                confidence

            )

        processing_time = round(

            time.time() - start,

            2

        )

        total_objects = sum(

            object_counts.values()

        )

        st.session_state.videos += 1

        st.session_state.objects += total_objects

        st.success("✅ Video Processed Successfully")

        st.video(output_video)

        st.markdown("---")

        st.subheader("📊 Video Analytics")

        v1, v2, v3, v4 = st.columns(4)

        with v1:
            st.metric(
                "Objects",
                total_objects
            )

        with v2:
            st.metric(
                "Classes",
                len(object_counts)
            )

        with v3:
            st.metric(
                "Tracking",
                "ByteTrack"
            )

        with v4:
            st.metric(
                "Time",
                f"{processing_time}s"
            )

        st.markdown("### 📦 Objects Found")

        if object_counts:

            for obj, count in object_counts.items():

                st.success(f"✅ {obj.title()} : {count}")

        else:

            st.warning("No objects detected.")

        report = "AI Vision Detection Report\n"
        report += "-" * 35 + "\n\n"

        for obj, count in object_counts.items():

            report += f"{obj.title()} : {count}\n"

        report += f"\nTotal Objects : {total_objects}"
        report += f"\nClasses : {len(object_counts)}"
        report += f"\nProcessing Time : {processing_time}s"

        st.download_button(

            "📄 Download Detection Report",

            report,

            file_name="detection_report.txt",

            mime="text/plain",

            use_container_width=True

        )

        with open(output_video, "rb") as file:

            st.download_button(

                "⬇ Download Processed Video",

                file,

                file_name="detected_video.mp4",

                mime="video/mp4",

                use_container_width=True

            )

st.markdown("---")
# ---------------- LIVE WEBCAM ---------------- #

if mode == "📷 Live Webcam":

    st.subheader("📷 Live AI Webcam Detection")

    start_camera = st.checkbox("Start Webcam")

    if start_camera:

        FRAME_WINDOW = st.image([])

        cap = cv2.VideoCapture(0)

        fps_placeholder = st.empty()

        object_placeholder = st.empty()

        start_time = time.time()

        total_frames = 0

        while start_camera:

            success, frame = cap.read()

            if not success:
                st.error("Unable to access webcam.")
                break

            results = model.track(
                frame,
                conf=confidence,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False
            )

            annotated = results[0].plot()

            FRAME_WINDOW.image(
                cv2.cvtColor(
                    annotated,
                    cv2.COLOR_BGR2RGB
                ),
                use_container_width=True
            )

            total_frames += 1

            elapsed = time.time() - start_time

            fps = total_frames / elapsed if elapsed > 0 else 0

            fps_placeholder.metric(
                "⚡ FPS",
                f"{fps:.1f}"
            )

            count = 0

            if results[0].boxes is not None:

                count = len(results[0].boxes)

            object_placeholder.metric(

                "📦 Objects",

                count

            )

        cap.release()
# ---------------- AI DASHBOARD ---------------- #

st.markdown("---")

st.subheader("📊 AI Vision Dashboard")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "📷 Images Processed",
        st.session_state.images
    )

with c2:
    st.metric(
        "🎥 Videos Processed",
        st.session_state.videos
    )

with c3:
    st.metric(
        "📦 Total Objects",
        st.session_state.objects
    )

with c4:
    st.metric(
        "🤖 AI Status",
        "🟢 Active"
    )

st.markdown("---")

# ---------------- ABOUT AI ---------------- #

st.subheader("🚀 AI Features")

feature1, feature2 = st.columns(2)

with feature1:

    st.success("""
✅ YOLOv8 Object Detection

✅ ByteTrack Tracking

✅ Image Detection

✅ Video Detection

✅ Live Webcam

✅ AI Analytics
""")

with feature2:

    st.info("""
⚡ Fast Processing

🎯 Adjustable Confidence

📊 Detection Summary

📄 Detection Report

⬇ Download Results

🤖 Deep Learning Powered
""")

st.markdown("---")

# ---------------- FOOTER ---------------- #

st.markdown(
"""
<div style='text-align:center;
padding:30px;
color:#A0AEC0;'>

<h2>🤖 AI Vision Object Detector Pro V4</h2>

<p>

Built using

<b>

Python • Streamlit • OpenCV • YOLOv8 • ByteTrack

</b>

</p>

<p>

Artificial Intelligence • Computer Vision • Object Detection

</p>

<p>

Developed by

<b>Vanshita Choudhary</b>

</p>

<p>

🚀 CodeAlpha Python Programming Internship Project

</p>

</div>
""",
unsafe_allow_html=True
)            