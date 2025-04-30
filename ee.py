import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Step 1: Define skin tone extraction function using MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

def extract_skin_tone(file_bytes):
    image = np.array(bytearray(file_bytes), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = image.shape
        cheek_coords = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in [93, 323]]
        cheek_colors = []

        for (x, y) in cheek_coords:
            x1, x2 = max(0, x - 10), min(image.shape[1], x + 10)
            y1, y2 = max(0, y - 10), min(image.shape[0], y + 10)
            region = image[y1:y2, x1:x2]
            if region.size > 0:
                avg_color = np.mean(region.reshape(-1, 3), axis=0)
                cheek_colors.append(avg_color)

        if cheek_colors:
            avg_skin_color = np.mean(cheek_colors, axis=0).astype(int)
            return avg_skin_color
    return None

# Step 2: Training Model
def train_outfit_model():
    data = {
        'skin_color': [
            [255, 224, 189], [255, 219, 172], [252, 194, 149], [242, 177, 121], [230, 160, 100],
            [210, 140, 90], [190, 130, 80], [170, 120, 70], [150, 110, 65], [130, 95, 60],
            [110, 85, 55], [90, 70, 50], [80, 60, 45], [70, 55, 40], [60, 45, 35],
            [255, 205, 148], [220, 185, 135], [200, 170, 120], [180, 155, 110], [160, 135, 100],
            [140, 115, 90], [120, 100, 80], [100, 80, 65], [85, 70, 55]
        ],
        'color_combination': [
            ["light pink", "sky blue", "ivory"], ["lavender", "mint", "white"], ["peach", "cream", "khaki"],
            ["coral", "navy blue", "white"], ["camel", "rust", "sage"], ["olive green", "beige", "denim"],
            ["gold", "cocoa", "teal"], ["burnt orange", "dark green", "brown"], ["plum", "olive", "grey"],
            ["mustard", "maroon", "charcoal"], ["terracotta", "cream", "forest green"],
            ["burgundy", "black", "tan"], ["eggplant", "midnight blue", "sand"],
            ["bronze", "deep teal", "grey"], ["black", "copper", "wine red"],
            ["blush pink", "denim", "ivory"], ["aqua", "sand", "light coral"], ["rosewood", "khaki", "cream"],
            ["beige", "hunter green", "slate blue"], ["dark brown", "moss green", "warm white"],
            ["mauve", "ash grey", "caramel"], ["deep blue", "chocolate", "bronze"],
            ["rich gold", "taupe", "eggplant"], ["black", "burgundy", "stone"]
        ]
    }

    df = pd.DataFrame(data)
    df['color_combination'] = df['color_combination'].apply(lambda x: ', '.join(x))
    le = LabelEncoder()
    df['encoded_colors'] = le.fit_transform(df['color_combination'])

    X = pd.DataFrame(df['skin_color'].tolist())
    y = df['encoded_colors']

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)

    joblib.dump(model, 'outfit_model.pkl')
    joblib.dump(le, 'label_encoder.pkl')

    return model, le

@st.cache_resource
def load_model():
    try:
        model = joblib.load('outfit_model.pkl')
        le = joblib.load('label_encoder.pkl')
    except:
        model, le = train_outfit_model()
    return model, le

# Step 3: Optional helper to convert named colors
def get_css_color(color_name):
    css_color_map = {
        "mint": "#98ff98", "ivory": "#fffff0", "sky blue": "#87ceeb", "peach": "#ffe5b4",
        "camel": "#c19a6b", "khaki": "#f0e68c", "rust": "#b7410e", "sage": "#9dc183",
        "denim": "#1560bd", "teal": "#008080", "plum": "#8e4585", "olive": "#808000",
        "mustard": "#ffdb58", "terracotta": "#e2725b", "burgundy": "#800020", "eggplant": "#614051",
        "midnight blue": "#191970", "sand": "#f4a460", "bronze": "#cd7f32", "deep teal": "#003f5c",
        "wine red": "#722f37", "hunter green": "#355e3b", "slate blue": "#6a5acd",
        "moss green": "#8a9a5b", "warm white": "#fefee2", "ash grey": "#b2beb5", "caramel": "#af6e4d",
        "chocolate": "#7b3f00", "taupe": "#483c32", "stone": "#8d8c87", "light pink": "#ffb6c1",
        "lavender": "#e6e6fa", "blush pink": "#ffcccc", "aqua": "#00ffff", "rosewood": "#65000b",
        "white": "#ffffff", "black": "#000000", "grey": "#808080", "brown": "#a52a2a"
    }
    return css_color_map.get(color_name.lower(), color_name)

# Step 4: Streamlit Interface
def main():
    st.set_page_config(page_title="Skin Tone Outfit Recommender", layout="centered")
    st.title('👗 Outfit Color Suggestion Based on Skin Tone')

    uploaded_file = st.file_uploader("Upload a face image (JPG)", type="jpg")
    
    if uploaded_file:
        file_bytes = uploaded_file.read()
        skin_tone = extract_skin_tone(file_bytes)

        if skin_tone is None:
            st.error("😕 No face detected in the image.")
            return

        st.image(file_bytes, caption="Uploaded Image", use_column_width=True)
        st.write(f"🧑 Detected Skin Color (RGB): {skin_tone.tolist()}")

        model, le = load_model()
        predicted_label = model.predict([skin_tone])
        suggested_combination = le.inverse_transform(predicted_label)[0]

        st.subheader("🎨 Suggested Color Combination:")
        st.write(suggested_combination)

        colors = suggested_combination.split(', ')
        for color in colors:
            hex_color = get_css_color(color)
            st.markdown(f"""
                <div style='display:flex; align-items:center; margin-bottom:10px'>
                    <div style='background-color:{hex_color}; width:60px; height:60px; margin-right:10px; border:1px solid #ccc'></div>
                    <span style='font-size:18px'>{color}</span>
                </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
