# Breast Cancer Prediction Using Multimodal Mammogram Data and Deep Learning

This project aims to develop a robust AI system for **early breast cancer detection** using a combination of private hospital datasets (Acibadem) and public sources (CBIS-DDSM, Mini-DDSM, Histopathology). The solution integrates **custom CNN architectures**, **pretrained models (VGG19, ResNet50, EfficientNetB0)**, and **hybrid models** using **multimodal data** to enhance accuracy and reduce diagnostic bias.

## 🚀 Key Features
- ✅ Data engineering from unstructured hospital metadata (19,000+ images)
- ✅ Unique image-label matching pipeline using metadata keys
- ✅ Multimodal model training using histopathology and mammography
- ✅ Performance comparison across CNN, VGG19, ResNet50, EfficientNet, and hybrid CNN+VGG models
- ✅ Implementation of class imbalance handling and data augmentation
- ✅ Clinical collaboration with Acibadem Hospital, Istanbul

## 📊 Results
- CNN + Histopathology + CBIS: **F1 = 0.88**
- ResNet50 + CLAHE preprocessing: **F1 = 0.90**
- Hybrid CNN + VGG16: **Accuracy = 92%**

## 💡 Methodologies
- Stratified cross-validation
- Data preprocessing: CLAHE, rotation, flipping, resizing
- Transfer learning with frozen and fine-tuned base layers
- Loss functions: Binary cross-entropy with class weights

## 🏥 Clinical Relevance
This system reduces diagnostic delays by supporting radiologists with AI-powered screening
