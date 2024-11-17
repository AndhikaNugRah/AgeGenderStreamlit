import time
import cv2 as cv
import numpy as np
import streamlit as st
from datetime import datetime

# Load the models
faceDetectArch = r"C:\Users\dhika\Downloads\AgeGender_Streamlit\Model\opencv_face_detect_arch.pbtxt"
faceDetectModel = r"C:\Users\dhika\Downloads\AgeGender_Streamlit\Model\opencv_face_detect_model.pb"

ageArch = r"C:\Users\dhika\Downloads\AgeGender_Streamlit\Model\age_arch.prototxt"
ageModel = r"C:\Users\dhika\Downloads\AgeGender_Streamlit\Model\age_model.caffemodel"

MFgenderArch = r"C:\Users\dhika\Downloads\AgeGender_Streamlit\Model\mf_gender_arch.prototxt"
MFgenderModel = r"C:\Users\dhika\Downloads\AgeGender_Streamlit\Model\mf_gender_model.caffemodel"

genderList = ['Male', 'Female']
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Load networks
ageNet = cv.dnn.readNet(ageModel, ageArch)
genderNet = cv.dnn.readNet(MFgenderModel, MFgenderArch)
faceNet = cv.dnn.readNet(faceDetectModel, faceDetectArch)
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, bboxes

# Streamlit UI
st.title("Age and Gender Recognition")
st.write("Upload an image or use the camera to detect age and gender.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv.imdecode(image, cv.IMREAD_COLOR)
    frameFace, bboxes = getFaceBox(faceNet, frame)

    if not bboxes:
        st.write("No face Detected in the uploaded image.")
    else:
        for bbox in bboxes:
            face = frame[max(0, bbox[1]):min(bbox[3], frame.shape[0] - 1), max(0, bbox[0]):min(bbox[2], frame.shape[1] - 1)]
            blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]

            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            ageCon = agePreds[0].max()
            age = ageList[agePreds[0].argmax()]

            label = "{}, {}".format(gender, age)

            text_size = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = bbox[0]
            text_y = bbox[1] - 10

            cv.rectangle(frameFace, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y + 5), (0, 0, 0), cv.FILLED)
            cv.putText(frameFace, label, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
        caption="Processed Image with Age:{} Confidence Score:{:.3f} and Gender:{}".format(age,ageCon,gender)
        st.image(frameFace, channels="BGR", caption=caption)

# Camera feature
def main():
    st.title("Age and Gender Detection")

    # Initialize session state for camera running status
    if 'camera_running' not in st.session_state:
        st.session_state.camera_running = False
    
    #Button to stop camera 
    if st.button("Stop Camera"):
        st.session_state.camera_running = False
        st.write("Camera stopped.")
    
    if st.button("Use Camera"):
        st.session_state.camera_running = True
        st.write("Starting camera...")
        
        cap = cv.VideoCapture(0) 
        padding = 20
        
        # Create a placeholder for the image
        image_placeholder = st.empty()

        while st.session_state.camera_running:
            # Read frame
            t = time.time()
            hasFrame, frame = cap.read()

            if not hasFrame:
                st.write("Failed to capture image.")
                break

            # Process the frame
            frameFace, bboxes = getFaceBox(faceNet, frame)

            if not bboxes:
                st.write("No face Detected, Checking next frame")
                continue
            best_age = None
            best_age_confidence = 0.0

            for bbox in bboxes:
                face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
                     max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]

                blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    
                # Gender prediction
                genderNet.setInput(blob)
                genderPreds = genderNet.forward()
                gender = genderList[genderPreds[0].argmax()]

                # Age prediction
                ageNet.setInput(blob)
                agePreds = ageNet.forward()
                age_confidence = agePreds[0].max()
                age = ageList[agePreds[0].argmax()]

                st.write("Gender: {}, Confidence: {:.3f}".format(gender, genderPreds[0].max()))


               # Check if the new age prediction confidence is higher than the previous best
                if age_confidence > best_age_confidence:
                    best_age_confidence = age_confidence
                    best_age = age

                # Display the best age prediction if it has been updated
                if best_age is not None:
                    st.write("Best Age: {}, Confidence: {:.3f}, Detected Time:{}".format(best_age, best_age_confidence, current_time))

                label = "{}, {}".format(gender, best_age if best_age is not None else "Unknown")
                text_size = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                text_x = bbox[0]
                text_y = bbox[1] - 10
                
                cv.rectangle(frameFace, (text_x, text_y - text_size[1] - 5), (text_x + text_size[0], text_y + 5), (0, 0, 0), cv.FILLED)
                cv.putText(frameFace, label, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)

            # Convert the frame to RGB for display in Streamlit 
            caption_text = "Live Camera Feed with Gender:{}, Age: {}, Confidence Age: {:.3f}, Detected at: {}".format(gender,best_age, best_age_confidence, current_time)
            frameFace = cv.cvtColor(frameFace, cv.COLOR_BGR2RGB)
            image_placeholder.image(frameFace, caption=caption_text, use_container_width=True)

            # Show processing time
            st.write("Processing time: {:.3f} seconds".format(time.time() - t))

if __name__ == "__main__":
    main()