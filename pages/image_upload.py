import torch
import streamlit as st
from PIL import Image
import io
import pandas as pd

# Model
st.title('Damaged Tree Detection')
model = torch.hub.load('./yolov5','custom',path = 'cocotree.pt',source ='local', force_reload =True)  # yolov5n - yolov5x6 official model
# 'custom', 'path/to/best.pt')  # custom model

st.write("Upload an image to predict the model")

# Images # or file, Path, URL, PIL, OpenCV, numpy, list
st.subheader('Input Image')
im = st.file_uploader('',type=['png', 'jpg', 'jpeg'], accept_multiple_files=False)
if im is not None:
    image = Image.open(im)
    buffered = io.BytesIO()
    image.save(buffered, quality=90, format='JPEG')
    st.image(image,caption='uploaded image')
    
    # Inference
    results = model(image)

    # Results
    st.subheader('Report')
    results.print()  # or .show(), .save(), .crop(), .pandas(), etc.

    results.pandas().xyxy[0]  # im predictions (pandas)
    #      xmin    ymin    xmax   ymax  confidence  class    name
    # 0  749.50   43.50  1148.0  704.5    0.874023      0  person
    # 2  114.75  195.75  1095.0  708.0    0.624512      0  person
    # 3  986.00  304.00  1028.0  420.0    0.286865     27     tie

    st.subheader('Number of Detections:')
    count=results.pandas().xyxy[0].value_counts('name')  # class counts (pandas)
    st.write(count)

    st.subheader('Predicted Image')
    img = results.render()
    im_rgb=img[0]
    st.image(im_rgb,caption='output')

    @st.experimental_memo
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')
    
    csv = convert_df(results.pandas().xyxy[0])
    st.download_button("Press to Download Results", csv, "file.csv", "text/csv", key='download-csv')



