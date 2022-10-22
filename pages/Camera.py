from logging import captureWarnings
import torch
import streamlit as st
from PIL import Image
import io
import pandas as pd

model = torch.hub.load('./yolov5','custom',path = 'cocotree.pt',source ='local', force_reload =True)

st.title('Damaged Tree Detection')
st.write('Take a picture to predict the model')

captured_image=st.camera_input('Take a snap')
if captured_image is None:
    st.write("Waiting for capture...")
else:
    st.write("Your input")
    st.image(captured_image)
    image = Image.open(captured_image)
    buffered = io.BytesIO()
    image.save(buffered, quality=90, format='JPEG')
    
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

    btn = st.download_button(
      label="Download image",
      data=captured_image,
      file_name="image.png",
      mime="image/png")

    @st.experimental_memo
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')
    
    csv = convert_df(results.pandas().xyxy[0])
    st.download_button("Press to Download Results", csv, "file.csv", "text/csv", key='download-csv')
