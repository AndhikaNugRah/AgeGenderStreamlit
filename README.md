This project implements an advanced age and gender detection system utilizing OpenCV and Streamlit, allowing users to analyze facial features in images and live video streams. 
The system employs a series of pre-trained models for accurate detection and classification.

Key Components:
Face Detection:

Architecture: 
The face detection model is defined by opencv_face_detect_arch.pbtxt and utilizes the opencv_face_detect_model.pb for identifying faces within images or video streams.

Age Prediction:
The age detection capability is powered by a model defined in age_arch.prototxt and utilizes the age_model.caffemodel to estimate the age of detected individuals within specified ranges.

Gender Prediction:
The gender detection functionality is facilitated by the mf_gender_arch.prototxt and mf_gender_model.caffemodel, which classify detected faces as male or female with high accuracy.

User Interaction:
The application provides a user-friendly interface through Streamlit, allowing users to upload images or utilize their device's camera for real-time analysis.
Upon uploading an image or activating the live camera, the system processes the input, detects faces, and predicts both age and gender, displaying the results directly on the interface.

Download Model Here:
https://drive.google.com/drive/folders/1Enun5MaFQmTaSsi22wbLrdwBLqa_qDsk?usp=sharing

This project showcases the integration of computer vision and web technologies, making age and gender detection accessible and interactive for users in various applications, such as demographic analysis, user profiling, and more.
