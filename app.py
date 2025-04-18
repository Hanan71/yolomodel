import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.title("🔍 كاشف الحجاج باستخدام YOLO")

model = YOLO("yolov5s.pt")

uploaded_file = st.file_uploader("ارفع صورة أو مقطع", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file:
    if uploaded_file.name.endswith(".mp4"):
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.read())
        st.video("temp_video.mp4")
        results = model.predict("temp_video.mp4")
        st.write("✅ تم التحليل، ولكن عرض الفيديو مع النتائج غير مدعوم مباشرة.")
    else:
        image = Image.open(uploaded_file)
        st.image(image, caption="الصورة الأصلية")
        results = model.predict(image)
        for result in results:
            st.image(result.plot(), caption="النتائج")
