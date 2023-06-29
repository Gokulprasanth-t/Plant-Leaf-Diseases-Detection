from flask import Flask,render_template,Response
import cv2
from ultralytics.yolo.engine.model import YOLO
import pyttsx3
import torch

engine=pyttsx3.init()

app=Flask(__name__)

camera=cv2.VideoCapture(1)
yolo = YOLO("D:/Hackathon/Innovsense/best.pt")

def generate_frames():
    class_label=[]
    while True:
        
        ret,frame = camera.read()


        detections = yolo.predict(source = frame ,conf=0.75,show=True)
        print(detections)

        if detections[0].boxes.shape[0] > 0:
            class_label=detections[0].boxes.cls
        
            sign=class_label.tolist() 
            audio(int(sign[0]))     


def audio(sign):
        
            if (sign==1):

                engine.say("Guava Algal Leaf Spot")
                engine.runAndWait()
                


            elif(sign==0):

                engine.say("Fig Leaf Galls")
                engine.runAndWait()
                
            

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace;boundary=frame')



if __name__=="__main__":
    app.run(debug=True)
