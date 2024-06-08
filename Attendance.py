import streamlit as st
import requests
import cv2
import numpy as np
import face_recognition
import os
from streamlit_lottie import st_lottie
from datetime import datetime

st.set_page_config(page_title="Face Recognition",layout="wide")

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code !=200:
        return None
    return r.json()


lottie_face=load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_qudievat.json")
with st.container():
   st.title("Face Recognition to mark attendance.")

with st.container():
    st.write("---")
    st_lottie(lottie_face, height=300)


run=st.checkbox('Run')
Frame_window=st.image([])
path='Attendanceimages'
images=[]
names=[]
list=os.listdir(path)


for c in list:
    currentimage=cv2.imread(f'{path}/{c}')
    images.append(currentimage)
    names.append(os.path.splitext(c)[0])



def findencoding(images):
    encodelist=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

def markattendance(name):
    with open('Attendance.csv', 'r+') as f:
        mydatalist= f.readlines()
        namelist=[]
        for line in mydatalist:
            entry=line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now=datetime.now()
            dtstring=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtstring}')

encodelistknown=findencoding(images)

cap=cv2.VideoCapture(0)

while run:
    success,img=cap.read()
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    imgsmall=cv2.resize(img,(0,0),None,0.25,0.25)
    imgsmall=cv2.cvtColor(imgsmall,cv2.COLOR_BGR2RGB)
    facescurrentframe=face_recognition.face_locations(imgsmall)
    encodecurrentframe=face_recognition.face_encodings(imgsmall,facescurrentframe)

    for encodeface,faceloc in zip(encodecurrentframe,facescurrentframe):
        match=face_recognition.compare_faces(encodelistknown,encodeface)
        faceDis=face_recognition.face_distance(encodelistknown,encodeface)
        matchIndex=np.argmin(faceDis)
        if match[matchIndex] :
              name=names[matchIndex].upper()
              y1,x2,y2,x1=faceloc
              y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
              cv2.rectangle(img,(x1,y1),(x2,y2),(204,0,102),2)
              cv2.rectangle(img,(x1,y1-30),(x2,y2),(204,0,102))
              cv2.putText(img,name,(x1+4,y1-4),cv2.FONT_ITALIC,1,(255,255,255),1)
              markattendance(name)

        Frame_window.image(img)


else:
    st.write('Stopped')




