import streamlit as st
from PIL import Image
import cv2
import numpy as np
import requests
import pymupdf
from docx import Document
from sklearn.cluster import KMeans

st.set_page_config(page_title="Vision Warriors", page_icon="🦾")

COLOR_NAMES = {
    "Black": (0, 0, 0),
    "White": (255, 255, 255),
    "Red": (255, 0, 0),
    "Green": (0, 128, 0),
    "Blue": (0, 0, 255),
    "Yellow": (255, 255, 0),
    "Orange": (255, 165, 0),
    "Purple": (128, 0, 128),
    "Pink": (255, 192, 203),
    "Brown": (139, 69, 19),
    "Gray": (128, 128, 128),
    "Cyan": (0, 255, 255),
    "Magenta": (255, 0, 255),
    "Light Gray": (211, 211, 211),
    "Dark Gray": (64, 64, 64),
    "Light Blue": (173, 216, 230),
    "Dark Blue": (0, 0, 139),
    "Light Green": (144, 238, 144),
    "Dark Green": (0, 100, 0),
}

def closest_color(rgb):
    min_dist = float("inf")
    name = "Unknown"
    for color_name, color_rgb in COLOR_NAMES.items():
        dist = np.linalg.norm(np.array(rgb) - np.array(color_rgb))
        if dist < min_dist:
            min_dist = dist
            name = color_name
    return name

def rgb_to_hex(rgb):
    return "#%02x%02x%02x" % tuple(int(x) for x in rgb)

def dominant_color(image, k=1):
    image = np.array(image.convert("RGB"))
    pixels = image.reshape(-1, 3)

    # Sampling keeps large images from using excessive memory.
    if len(pixels) > 10000:
        indices = np.random.choice(len(pixels), 10000, replace=False)
        pixels = pixels[indices]

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(pixels)
    cluster = np.argmax(np.bincount(kmeans.labels_))
    return tuple(kmeans.cluster_centers_[cluster].astype(int))

def image_average_color(image):
    image = np.array(image.convert("RGB"))
    return tuple(np.mean(image, axis=(0, 1)).astype(int))

def ocr_image(image_bytes):
    api_key = st.secrets.get("OCR_SPACE_API_KEY")

    if not api_key:
        return "OCR is not configured. Add OCR_SPACE_API_KEY to Streamlit secrets."

    response = requests.post(
        "https://api.ocr.space/parse/image",
        files={"file": ("image.png", image_bytes)},
        data={
            "isOverlayRequired": False,
            "apikey": api_key,
            "language": "eng",
        },
        timeout=60,
    )
    response.raise_for_status()
    result = response.json()

    try:
        return result["ParsedResults"][0]["ParsedText"]
    except (KeyError, IndexError):
        return "Could not extract text from this image."

def extract_pdf(file_bytes):
    doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    text = ""
    colors = []

    for page in doc:
        text += page.get_text("text")

        for block in page.get_text("dict")["blocks"]:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    if "color" in span:
                        color_int = span["color"]
                        r = (color_int >> 16) & 255
                        g = (color_int >> 8) & 255
                        b = color_int & 255
                        colors.append((r, g, b))

    avg_color = (
        tuple(np.mean(colors, axis=0).astype(int))
        if colors else (0, 0, 0)
    )
    return text, avg_color

def extract_docx(file_bytes):
    import io

    doc = Document(io.BytesIO(file_bytes))
    text = ""
    colors = []

    for para in doc.paragraphs:
        for run in para.runs:
            text += run.text
            if run.font.color and run.font.color.rgb:
                rgb = run.font.color.rgb
                colors.append((rgb[0], rgb[1], rgb[2]))
        text += "\n"

    avg_color = (
        tuple(np.mean(colors, axis=0).astype(int))
        if colors else (0, 0, 0)
    )
    return text, avg_color

st.title("🦾 Vision Warriors")
st.write("Accessibility tools for extracting text and identifying colors.")

tab1, tab2 = st.tabs(["📄 File Reader", "🎥 Camera Color Detector"])

with tab1:
    uploaded = st.file_uploader(
        "Upload a PDF, TXT, DOCX, or image",
        type=["pdf", "txt", "docx", "png", "jpg", "jpeg"],
    )

    if uploaded:
        file_bytes = uploaded.getvalue()
        file_type = uploaded.name.lower()

        try:
            if file_type.endswith((".png", ".jpg", ".jpeg")):
                image = Image.open(uploaded).convert("RGB")
                st.image(image, caption="Uploaded image", width="stretch")

                image_array = np.array(image)

                h, w = image_array.shape[:2]

                roi = image_array[
                    h // 2 - 50:h // 2 + 50,
                    w // 2 - 50:w // 2 + 50
                ]

                color = tuple(np.mean(roi, axis=(0, 1)).astype(int))
                text = ocr_image(file_bytes)

            elif file_type.endswith(".pdf"):
                text, color = extract_pdf(file_bytes)

            elif file_type.endswith(".docx"):
                text, color = extract_docx(file_bytes)

            elif file_type.endswith(".txt"):
                text = file_bytes.decode("utf-8")
                color = (0, 0, 0)

            else:
                text = "Unsupported file type."
                color = (0, 0, 0)

            color_name = closest_color(color)
            hex_color = rgb_to_hex(color)

            st.subheader("Detected Color")
            st.markdown(
                f'<div style="background-color:{hex_color};'
                f'padding:20px;border-radius:10px;color:black">'
                f'<b>{color_name}</b> — RGB {color}</div>',
                unsafe_allow_html=True,
            )

            st.subheader("Extracted Text")
            st.text_area("Text", text, height=350)

        except Exception as e:
            st.error(f"Error processing file: {e}")

with tab2:
    camera_image = st.camera_input("Take a picture")

    if camera_image:
        image = Image.open(camera_image).convert("RGB")
        st.image(image, caption="Camera image", width="stretch")



        image_array = np.array(image)

        h, w = image_array.shape[:2]

        roi = image_array[
            h // 2 - 50:h // 2 + 50,
            w // 2 - 50:w // 2 + 50
        ]

        color = tuple(np.mean(roi, axis=(0, 1)).astype(int))
        
        color_name = closest_color(color)
        hex_color = rgb_to_hex(color)

        st.subheader("Detected Color")
        st.markdown(
            f'<div style="background-color:{hex_color};'
            f'padding:30px;border-radius:10px;color:black">'
            f'<h2>{color_name}</h2>'
            f'<p>RGB: {color}</p></div>',
            unsafe_allow_html=True,
        )
