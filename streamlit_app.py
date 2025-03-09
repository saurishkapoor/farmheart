from ultralytics import YOLO
import streamlit as st
from PIL import Image
import streamlit_shadcn_ui as ui

st.set_page_config(page_title="FarmHeart", page_icon="Farmheart.png", layout="centered", initial_sidebar_state="auto", menu_items=None)

#Load model
@st.cache_resource
def models():
	mod = YOLO('cnn.pt')
	return mod

personal = "https://www.saurishkapoor.com"

st.image("farmheartbanner2.svg")

nav=st.sidebar.radio("Navigation",["Home", "How it Works","Diagnosis"] )

if nav == "Home":  
    st.logo("Farmheart.png")
    st.subheader("Problem")
    st.write("Cardiovascular disease (CVD) affects up to 6.7% of cattle, yet remains underdiagnosed due to the reliance on ECG, echocardiography, and biomarker tests, which cost \$45–$100 per animal and require specialized veterinary expertise. Delayed detection leads to up to 30% losses in dairy and meat productivity, increased morbidity, and higher treatment costs. ")
    st.image("cattle.jpg")
    st.subheader("Solution")
    st.write("We propose a novel deep learning-augmented retinal imaging system for non-invasive CVD diagnosis in cattle. By analyzing fundus images of the retina with a convolutional neural network (CNN) trained on vascular biomarkers (e.g., tortuosity, hemorrhage), Farmheart enables farmers to upload images for an instant AI-driven diagnosis. This approach can help reduce costs of diagnosising cattle with CVD to near-zero after deployment and cuts diagnostic time from several days to seconds.")
    
    st.markdown("")
    with ui.element("div", className="flex gap-2", key="buttons_group1"):
        ui.element("link_button", text="About Saurish", url="https://spangled-viscose-952.notion.site/Saurish-Kapoor-1476de9535198000b67ecba4f2fba117", key="btn1")
        ui.element("link_button", text="Github", url="https://github.com/saurishkapoor/farmheart", variant="outline", key="btn2")

elif nav == "How it Works":
    st.logo("Farmheart.png")
    st.subheader("Dataset")
    st.markdown("- The RGB retinal images that constitute the dataset were captured using the Optomed Smartscope digital fundus camera.")
    st.markdown("- A total of 1,118 images with a resolution of 1536x1152 pixel in JPG were collected from the right and left eyes of 100 cattle (52 had CVD and 48 were non-CVD) from an online opensource dataset available on Kaggle.")
    st.markdown("- After extraction of the raw data, cleaning took place - the code can be found here, and the dataset we utelized can be found here, respectivley")
    st.markdown('''
    <style>
    [data-testid="stMarkdownContainer"] ul{
        padding-left:40px;
    }
    </style>
    ''', unsafe_allow_html=True)

    st.subheader("Detection")
    st.write("To detect CVD in cattle through retinal images like the ones shown below, several physiological biomarkers can be analyzed:")
    st.markdown("- Vessel Morphology")
    st.markdown("- Vessel Caliber (Thickness)")
    st.markdown("- Retinal Hemorrhages")
    st.markdown("- Arteriovenous Ratio (AVR)")
    st.markdown("- Color and Reflections")
    st.markdown("- Focal Lesions or Microaneurysms")
    st.markdown('''
    <style>
    [data-testid="stMarkdownContainer"] ul{
        padding-left:40px;
    }
    </style>
    ''', unsafe_allow_html=True)
    st.image("sample.jpg", caption="Samples of CVD and Non-CVD retinal images")

elif nav == "Diagnosis":
    st.logo("Farmheart.png")
    with st.form("option_1_form"):
        name = st.text_input("Name")
        email = st.text_input("Email")
        image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
        submit_button = st.form_submit_button("Submit")
        
        if submit_button:
            st.write("Form submitted", name, "!")
            if image:
                img = Image.open(image)
                st.image(img, caption="Uploaded image")
                model = models()
                res = model.predict(img)
                label = res[0].probs.top5
                conf = res[0].probs.top5conf
                conf = conf.tolist()
                label = str(res[0].names[label[0]].title())
                score = str(conf[0])
                st.write("Prediction:", label) 
                st.write("Confidence:", score)
