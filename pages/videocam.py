import torch
import streamlit as st
from PIL import Image
import base64
import io
import pandas as pd
import cv2

st.title('Damaged Tree Detection')
st.write('Please check the checkbox present in the sidebar to turn on your cam and uncheck the same to turn off')
model = torch.hub.load('./yolov5','custom',path = 'cocotree.pt',source ='local', force_reload =True)

FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(1)
run = st.sidebar.checkbox('Run')
cur_frame=0
frame_skip=40
total_count=0


while run:
    _, frame = camera.read()
    if cur_frame % frame_skip == 0:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)
        pil_img = Image.fromarray(frame) # convert opencv frame (with type()==numpy) into PIL Image

        results = model(pil_img)

        count=results.pandas().xyxy[0].value_counts('name')  # class counts (pandas)
        if count.empty:
            pass
        else:
            st.image(pil_img)
            st.subheader('Report')
            results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
            results.pandas().xyxy[0]  # im predictions (pandas)
                
            st.subheader('Number of Detections:')
            total_count+=count
            st.write(count)

            st.subheader('Predicted Image')
            img = results.render()
            im_rgb=img[0]
            st.image(im_rgb,caption='output')
            st.markdown("""---""")
    cur_frame+=1
