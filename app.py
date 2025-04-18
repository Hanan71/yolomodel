import streamlit as st
from ultralytics import YOLO
from PIL import Image
import streamlit.components.v1 as components

# إعداد صفحة Streamlit
st.set_page_config(page_title="YOLO Camera App", layout="centered")
st.title("🔍 كاشف الحجاج باستخدام YOLO")

# تشغيل الكاميرا في المتصفح
st.subheader("📸 جرب بنفسك")

camera_html = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Camera</title>
  <style>
    body {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 20px;
      font-family: Arial, sans-serif;
    }

    video {
      width: 100%;
      max-width: 600px;
      border: 2px solid #ccc;
      border-radius: 10px;
    }

    .controls {
      margin-top: 10px;
    }

    button {
      padding: 10px 20px;
      font-size: 16px;
      margin: 0 10px;
      cursor: pointer;
    }
  </style>
</head>
<body>
  <h3>Try it Yourself 💡</h3>
  
  <video id="video" autoplay playsinline></video>
  
  <div class="controls">
    <button onclick="startCamera()">Start</button>
    <button onclick="stopCamera()">Stop</button>
  </div>

  <script>
    let video = document.getElementById('video');
    let stream;

    async function startCamera() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
      } catch (err) {
        alert("Camera permission denied!");
        console.error(err);
      }
    }

    function stopCamera() {
      if (stream) {
        let tracks = stream.getTracks();
        tracks.forEach(track => track.stop());
        video.srcObject = null;
      }
    }
  </script>
</body>
</html>
"""

components.html(camera_html, height=600)

# قسم YOLO لتحليل صورة أو فيديو
st.subheader("📂 أو ارفع صورة / فيديو للتحليل")

model = YOLO("yolov5s.pt")

uploaded_file = st.file_uploader("ارفع صورة أو مقطع", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file:
    if uploaded_file.name.endswith(".mp4"):
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.read())
        st.video("temp_video.mp4")
        results = model.predict("temp_video.mp4")
        st.success("✅ تم التحليل، ولكن عرض النتائج مباشرة غير مدعوم.")
    else:
        image = Image.open(uploaded_file)
        st.image(image, caption="الصورة الأصلية")
        results = model.predict(image)
        for result in results:
            st.image(result.plot(), caption="النتائج")
