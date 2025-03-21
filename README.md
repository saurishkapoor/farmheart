# 🩺 AI-Powered Cattle CVD Detection  

## 🚨 Problem  
Cardiovascular disease (CVD) in cattle is **difficult to diagnose** due to its **low incidence** and the **limitations of traditional methods** (ECG, echocardiography, and blood biomarkers). Current techniques are often:  
- **Expensive** 💰 – Require specialized equipment and trained professionals.  
- **Time-Consuming** ⏳ – Diagnosis is often delayed until symptoms worsen.  
- **Invasive** 🩸 – Blood tests and other physical exams can be stressful for the animal.  
These challenges lead to **late detection**, poor prognosis, and economic losses for farmers.  

## 💡 Solution  
This project introduces an **AI-driven, non-invasive diagnostic tool** that uses **retinal fundus imaging** and **deep learning (YOLOv8n)** to detect CVD in cattle.  

🔹 **How It Works:**  
1️⃣ Farmers/Veterinarians upload a **retinal image** of cattle.  
2️⃣ The **YOLOv8n model** analyzes vascular features in the retina.  
3️⃣ The app provides an **instant CVD/Non-CVD diagnosis** with a confidence score.  

### 🎯 Key Features:  
✅ **Non-Invasive:** Uses eye imaging instead of blood tests.  
✅ **Fast & Accessible:** Instant results through a **user-friendly web app**.  
✅ **Cost-Effective:** Reduces reliance on expensive equipment.  
✅ **Scalable:** Can be extended to **other livestock and even human applications**.  

## 📌 Results  
🔹 **Accuracy:** 77.5%  
🔹 **Sensitivity:** 69%  
🔹 **Specificity:** 85%  
🔹 **F1-Score:** 75.1%  

## 🛠️ Tech Stack  
- **Model:** YOLOv8n (Ultralytics)  
- **Preprocessing:** CLAHE (Contrast Enhancement), OpenCV  
- **Web App:** Streamlit + Streamlit-Shadcn-UI  
- **Programming Language:** Python  

## 👤 About Me  
Hey there! 👋 I’m **Saurish Kapoor**, a high school researcher passionate about AI, biomedical engineering, and healthcare innovation. I built this project to bridge the gap between **AI research and real-world applications**, making veterinary diagnostics **more accessible and efficient**.  

🔗 **Connect with me:**  
📩 [LinkedIn](https://www.linkedin.com/in/saurishkapoor) | 🐦 [Twitter](https://x.com/_saurish)  

If you're interested in collaborating or improving this project, feel free to contribute. 💡✨  
