import os
import numpy as np
import streamlit as st
from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img


model = load_model("FV.h5")

labels = {
    0: "apple", 1: "banana", 2: "beetroot", 3: "bell pepper", 4: "cabbage", 5: "capsicum", 6: "carrot",
    7: "cauliflower", 8: "chilli pepper", 9: "corn", 10: "cucumber", 11: "eggplant", 12: "garlic", 13: "ginger",
    14: "grapes", 15: "jalepeno", 16: "kiwi", 17: "lemon", 18: "lettuce",
    19: "mango", 20: "onion", 21: "orange", 22: "paprika", 23: "pear", 24: "peas", 25: "pineapple",
    26: "pomegranate", 27: "potato", 28: "raddish", 29: "soy beans", 30: "spinach", 31: "sweetcorn",
    32: "sweetpotato", 33: "tomato", 34: "turnip", 35: "watermelon"
}

vegetables = [
    "Beetroot", "Cabbage", "Capsicum", "Carrot", "Cauliflower", "Corn", "Cucumber", "Eggplant", "Ginger",
    "Lettuce", "Onion", "Peas", "Potato", "Raddish", "Soy Beans", "Spinach", "Sweetcorn", "Sweetpotato",
    "Tomato", "Turnip"
]

def fetch_calories(prediction):
    exact_calories = {
        "apple": "52 calories", "banana": "89 calories", "beetroot": "43 calories", "bell pepper": "20 calories",
        "cabbage": "25 calories", "capsicum": "40 calories", "carrot": "41 calories", "cauliflower": "25 calories",
        "chilli pepper": "40 calories", "corn": "86 calories", "cucumber": "15 calories", "eggplant": "25 calories",
        "garlic": "149 calories", "ginger": "80 calories", "grapes": "69 calories", "jalepeno": "29 calories",
        "kiwi": "61 calories", "lemon": "29 calories", "lettuce": "15 calories", "mango": "60 calories",
        "onion": "40 calories", "orange": "47 calories", "paprika": "282 calories", "pear": "57 calories",
        "peas": "81 calories", "pineapple": "50 calories", "pomegranate": "83 calories", "potato": "77 calories",
        "raddish": "16 calories", "soy beans": "173 calories", "spinach": "23 calories", "sweetcorn": "86 calories",
        "sweetpotato": "86 calories", "tomato": "18 calories", "turnip": "28 calories", "watermelon": "30 calories"
    }
    food_item = prediction.lower().strip()
    return exact_calories.get(food_item, "Calorie data unavailable")

def processed_img(img_path):
    img = load_img(img_path, target_size=(224, 224, 3))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    answer = model.predict(img)
    y_class = answer.argmax(axis=-1)
    y = int(" ".join(str(x) for x in y_class))
    return labels[y].capitalize()

def run():
    st.title("AI Smart Nutrition Analyzer 🧠🥦")
    img_file = st.file_uploader("Upload a Fruit or Vegetable Image", type=["jpg", "png"])

    if img_file is not None:
        img = Image.open(img_file).resize((250, 250))
        st.image(img, width=250)

        if not os.path.exists("./upload_images"):
            os.makedirs("./upload_images")

        save_image_path = "./upload_images/" + img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        result = processed_img(save_image_path)

        if result in vegetables or result.lower() in [v.lower() for v in vegetables]:
            st.info("**Category : Vegetable**")
        else:
            st.info("**Category : Fruit**")

        st.success("**Predicted Item: " + result + "**")

        cal = fetch_calories(result)
        if cal:
            st.warning("**Estimated Energy: " + cal + " (Per 100 grams)**")

if __name__ == "__main__":
    run()
