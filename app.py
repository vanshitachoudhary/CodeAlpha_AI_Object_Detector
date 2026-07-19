import streamlit as st
import numpy as np
from PIL import Image
from detector import detect_image, detect_video, get_model
import plotly.express as px
import pandas as pd
import time
import os

# ---------------- PAGE CONFIG ---------------- #

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

model = get_model()

# ---------------- CSS ---------------- #

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html,body,[class*="css"]{
font-family:'Poppins',sans-serif;
}

.main{
background:linear-gradient(135deg,#0f172a,#111827,#1e293b);
color:white;
}

.block-container{
padding-top:2rem;
max-width:1400px;
}

section[data-testid="stSidebar"]{
background:#111827;
border-right:1px solid #374151;
}

.hero{

background:linear-gradient(90deg,#2563eb,#7c3aed);

padding:35px;

border-radius:20px;

color:white;

box-shadow:0px 10px 40px rgba(0,0,0,.35);

margin-bottom:25px;

}

.card{

background:#1f2937;

padding:18px;

border-radius:18px;

text-align:center;

box-shadow:0px 8px 25px rgba(0,0,0,.35);

transition:.3s;

}

.card:hover{

transform:translateY(-5px);

}

.metric-card{

background:#111827;

padding:20px;

border-radius:18px;

text-align:center;

border:1px solid #374151;

}

</style>
""", unsafe_allow_html=True)

# ---------------- HERO ---------------- #

st.markdown("""

<div class="hero">

<h1>🤖 AI Vision Object Detector Pro</h1>

<h4>YOLOv8 • ByteTrack • OpenCV • Deep Learning</h4>

<p>
Detect objects in Images, Videos and Live Webcam with AI.
</p>

</div>

""", unsafe_allow_html=True)

# ---------------- SIDEBAR ---------------- #

st.sidebar.title("🤖 AI Vision")

st.sidebar.markdown("### Version 5.0")

st.sidebar.success("🟢 AI Online")

mode = st.sidebar.radio(

"Detection Mode",

[
"🖼 Image Detection",
"🎥 Video Detection",
"📷 Live Webcam"
]

)

confidence = st.sidebar.slider(

"Confidence",

0.10,

1.00,

0.40,

0.05

)

st.sidebar.markdown("---")

st.sidebar.subheader("AI Information")

st.sidebar.write("🤖 Model : YOLOv8")

st.sidebar.write("🎯 Tracking : ByteTrack")

st.sidebar.write("💻 Python : 3.11")

st.sidebar.write("⚡ Status : Active")

# ---------------- DASHBOARD ---------------- #

c1,c2,c3,c4 = st.columns(4)

with c1:
    st.metric(
        "🖼 Images",
        st.session_state.images
    )

with c2:
    st.metric(
        "🎥 Videos",
        st.session_state.videos
    )

with c3:
    st.metric(
        "📦 Objects",
        st.session_state.objects
    )

with c4:
    st.metric(
        "🤖 AI",
        "Online"
    )

st.markdown("---")
# ---------------- IMAGE DETECTION ---------------- #

if mode == "🖼 Image Detection":

    st.subheader("🖼 AI Image Detection")

    uploaded_image = st.file_uploader(
        "Upload an Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image:

        start = time.time()

        image = Image.open(uploaded_image).convert("RGB")

        image_np = np.array(image)

        with st.spinner("🤖 AI is analyzing your image..."):

            detected_image, object_counts = detect_image(
                image_np,
                confidence
            )

        end = time.time()

        process_time = round(end - start, 2)

        st.session_state.images += 1

        total_objects = sum(object_counts.values())

        st.session_state.objects += total_objects

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("### 📷 Original")

            st.image(
                image,
                use_container_width=True
            )

        with col2:

            st.markdown("### 🎯 Detection Result")

            st.image(
                detected_image,
                use_container_width=True
            )

        st.markdown("---")

        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.metric(
                "📦 Objects",
                total_objects
            )

        with m2:
            st.metric(
                "🏷 Classes",
                len(object_counts)
            )

        with m3:
            st.metric(
                "🎯 Confidence",
                f"{int(confidence*100)}%"
            )

        with m4:
            st.metric(
                "⚡ Time",
                f"{process_time}s"
            )

        st.markdown("## 📊 Detection Analytics")

        if object_counts:

            df = pd.DataFrame(
                {
                    "Object": list(object_counts.keys()),
                    "Count": list(object_counts.values())
                }
            )

            chart1, chart2 = st.columns(2)

            with chart1:

                fig = px.bar(
                    df,
                    x="Object",
                    y="Count",
                    text="Count",
                    title="Detected Objects"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            with chart2:

                fig2 = px.pie(
                    df,
                    names="Object",
                    values="Count",
                    hole=.45,
                    title="Object Distribution"
                )

                st.plotly_chart(
                    fig2,
                    use_container_width=True
                )

            st.markdown("### 📋 Detection Summary")

            for obj, count in object_counts.items():

                st.success(f"✅ {obj} : {count}")

            csv = df.to_csv(index=False)

            st.download_button(
                "📄 Download Detection Report",
                csv,
                file_name="detection_report.csv",
                mime="text/csv",
                use_container_width=True
            )

        else:

            st.warning("No objects detected.")
# ---------------- VIDEO DETECTION ---------------- #

if mode == "🎥 Video Detection":

    st.subheader("🎥 AI Video Detection")

    uploaded_video = st.file_uploader(
        "Upload a Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video:

        start = time.time()

        with st.spinner("🤖 AI is processing your video..."):

            output_video, object_counts = detect_video(
                uploaded_video,
                confidence
            )

        end = time.time()

        process_time = round(end-start,2)

        st.session_state.videos += 1
        st.session_state.objects += sum(object_counts.values())

        st.success("✅ Video Processed Successfully")

        st.video(output_video)

        st.markdown("---")

        c1,c2,c3,c4 = st.columns(4)

        with c1:
            st.metric("📦 Objects",sum(object_counts.values()))

        with c2:
            st.metric("🏷 Classes",len(object_counts))

        with c3:
            st.metric("⚡ Time",f"{process_time}s")

        with c4:
            st.metric("🎯 Confidence",f"{int(confidence*100)}%")

        if object_counts:

            df = pd.DataFrame({

                "Object":list(object_counts.keys()),
                "Count":list(object_counts.values())

            })

            fig = px.bar(
                df,
                x="Object",
                y="Count",
                text="Count",
                title="Detected Objects"
            )

            st.plotly_chart(fig,use_container_width=True)

            st.markdown("### 📋 Detection Summary")

            for obj,count in object_counts.items():

                st.success(f"✅ {obj} : {count}")

        with open(output_video,"rb") as file:

            st.download_button(

                "⬇ Download Processed Video",

                file,

                file_name="detected_video.mp4",

                mime="video/mp4",

                use_container_width=True

            )

# ---------------- WEBCAM ---------------- #

if mode=="📷 Live Webcam":

    st.subheader("📷 Live AI Webcam")

    run = st.checkbox("Start Webcam")

    frame_window = st.image([])

    if run:

        import cv2

        cap = cv2.VideoCapture(0)

        while run:

            success,frame = cap.read()

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

            frame_window.image(

                cv2.cvtColor(

                    annotated,

                    cv2.COLOR_BGR2RGB

                )

            )

        cap.release()

# ---------------- FINAL DASHBOARD ---------------- #

st.markdown("---")

st.subheader("📊 AI Vision Dashboard")

d1,d2,d3,d4 = st.columns(4)

with d1:
    st.metric("Images",st.session_state.images)

with d2:
    st.metric("Videos",st.session_state.videos)

with d3:
    st.metric("Objects",st.session_state.objects)

with d4:
    st.metric("AI Status","🟢 Active")

st.markdown("---")

st.info("""

🤖 **AI Vision Object Detector Pro**

✅ Image Detection

✅ Video Detection

✅ Live Webcam

✅ YOLOv8 Deep Learning

✅ ByteTrack Tracking

✅ Interactive Analytics

✅ CSV Report Download

✅ Modern AI Dashboard

""")

st.markdown(
"""
<div style="text-align:center;padding:25px">

<h3>🤖 AI Vision Object Detector Pro v5</h3>

Built using Python • Streamlit • YOLOv8 • OpenCV • ByteTrack

<br><br>

Developed by <b>Vanshita Choudhary</b>

<br>

🚀 CodeAlpha Python Programming Internship

</div>

""",
unsafe_allow_html=True
)            