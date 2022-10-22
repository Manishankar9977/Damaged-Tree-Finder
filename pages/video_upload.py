import torch
import streamlit as st
import cv2
from PIL import Image
import pandas as pd

st.title('Damaged Tree Detection')
model = torch.hub.load('./yolov5','custom',path = 'cocotree.pt',source ='local', force_reload =True)
model.conf = 0.5

st.write('Upload a video to see predictions')

data=pd.DataFrame(columns=["xmin","ymin","xmax","ymax","confidence","class","name","Frame"])

uploaded_video = st.file_uploader("Choose video", type=["mp4", "mov"])
frame_skip = 40 # display every 300 frames

if uploaded_video is not None: # run only when user uploads video
    vid = uploaded_video.name
    with open(vid, mode='wb') as f:
        f.write(uploaded_video.read()) # save video to disk

    st.markdown(f"""
    ### Files
    - {vid}
    """,
    unsafe_allow_html=True) # display file name

    vidcap = cv2.VideoCapture(vid) # load video from disk
    cur_frame = 0
    success = True

    while success:
        success, frame = vidcap.read() # get next frame from video
        if cur_frame % frame_skip == 0: # only analyze every n=300 frames
            print('frame: {}'.format(cur_frame)) 
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame) # convert opencv frame (with type()==numpy) into PIL Image

            results = model(pil_img)

            count=results.pandas().xyxy[0].value_counts('name')  # class counts (pandas)
            if count.empty:
                pass
            else:
                st.subheader(cur_frame)
                st.image(pil_img)
                st.subheader('Report')
                results.print()  # or .show(), .save(), .crop(), .pandas(), etc.

                results.pandas().xyxy[0]  # im predictions (pandas)
                #      xmin    ymin    xmax   ymax  confidence  class    name
                # 0  749.50   43.50  1148.0  704.5    0.874023      0  person
                # 2  114.75  195.75  1095.0  708.0    0.624512      0  person
                # 3  986.00  304.00  1028.0  420.0    0.286865     27     tie
                A=results.pandas().xyxy[0]
                A['Frame']=cur_frame
                data = pd.concat([data, A], axis=0)

                st.subheader('Number of Detections:')
                st.write(count)

                st.subheader('Predicted Image')
                img = results.render()
                im_rgb=img[0]
                st.image(im_rgb,caption='output')
                st.markdown("""---""")
        cur_frame += 1
    st.write('Final Report')
    data
    @st.experimental_memo
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')
    
    csv = convert_df(data)
    st.download_button("Press to Download Results", csv, "file.csv", "text/csv", key='download-csv')
    st.markdown("<a href='#linkto_top'>Link to top</a>", unsafe_allow_html=True)
