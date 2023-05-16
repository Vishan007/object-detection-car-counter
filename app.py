import streamlit as st

original_title = '<p style="font-family:Helvetica; text-align: center; color:Blue; font-size: 45px;">Car Counter</p>'
st.markdown(original_title, unsafe_allow_html=True)

video = r'data\trafficlow.mp4'
video_file = open(video ,'rb')
st.video(video)

if st.button('Count car'):
    video = r'data\countedcar.mp4'
    video_file = open(video , 'rb')
    st.video(video)