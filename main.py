##########################  collecting face   ######################
import cv2
import numpy as numpy
import os
# Load HAAR face classifier
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
os.mkdir(f'./faces/narayan')
# Load functions
def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if faces is ():
        return None
    
    # Crop all faces found
    for (x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face

# Initialize Webcam
cap = cv2.VideoCapture(0)
count = 0

# Collect 100 samples of your face from webcam input
while True:

    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame), (200, 200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Save file in specified directory with unique name
        file_name_path = f'./faces/narayan/' + str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)

        # Put count on images and display live count
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Face Cropper', face)
        
    else:
        print("Face not found")
        pass

    if cv2.waitKey(1) == 13 or count == 100: #13 is the Enter Key
        break
        
cap.release()
cv2.destroyAllWindows()      
print("Collecting Samples Complete")

################################train the model for narayan face#########################################
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

# Get the training data we previously made
data_path = './faces/narayan/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

# Create arrays for training data and labels
Training_Data, Labels = [], []

# Open training images in our datapath
# Create a numpy array for training data
for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

# Create a numpy array for both training data and labels
Labels = np.asarray(Labels, dtype=np.int32)

# Initialize facial recognizer
# model = cv2.face.createLBPHFaceRecognizer()
# NOTE: For OpenCV 3.0 use cv2.face.createLBPHFaceRecognizer()
# pip install opencv-contrib-python
# model = cv2.createLBPHFaceRecognizer()

narayan_model  = cv2.face_LBPHFaceRecognizer.create()
# Let's train our model 
narayan_model.train(np.asarray(Training_Data), np.asarray(Labels))
print("Model trained sucessefully")

################## Train the model for my friend monil face##############################
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

# Get the training data we previously made
data_path = './faces/monil/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

# Create arrays for training data and labels
Training_Data, Labels = [], []

# Open training images in our datapath
# Create a numpy array for training data
for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

# Create a numpy array for both training data and labels
Labels = np.asarray(Labels, dtype=np.int32)

# Initialize facial recognizer
# model = cv2.face.createLBPHFaceRecognizer()
# NOTE: For OpenCV 3.0 use cv2.face.createLBPHFaceRecognizer()
# pip install opencv-contrib-python
# model = cv2.createLBPHFaceRecognizer()

monil_model  = cv2.face_LBPHFaceRecognizer.create()
# Let's train our model 
monil_model.train(np.asarray(Training_Data), np.asarray(Labels))
print("Model trained sucessefully")



###############  function to send whatsapp message  #####################

import pywhatkit as pkt
import datetime as dt
def sendWhatsAppMessage(mob,message):
    hour=int(dt.datetime.now().hour)
    minute=int(dt.datetime.now().minute)+1
    pkt.sendwhatmsg(mob,message,hour,minute)

##################   function to send email #########################
import smtplib
def sendMail(sender,receiver,message):
    server=smtplib.SMTP('smtp.gmail.com',587)
    server.starttls()
    server.login(sender,'dqetzjsjrivkcpvh')
    server.sendmail(sender,receiver,message)

#################  function to create ec2 instance and attach ebs volume using terraform  ##############
from subprocess import getstatusoutput
def ec2_ebs(key_name,region,ami,instance_type,ebs_size):
    print("Creating Infrastructure....")
    status,output=getstatusoutput(f'terraform -chdir="./terraform_aws" init && \
    terraform -chdir="./terraform_aws" apply \
    -var key_name={key_name} \
    -var ami={ami} \
    -var instance_type={instance_type} \
    -var ebs_size={ebs_size} \
    -var region={region}\
    -auto-approve ')
    if status==0:
        return "Infrastructure created successfully.."
    else:
        return "Infrastructure created successfully.."


import cv2
import numpy as np
import os

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
###########################   creating  function to extract face from image###################################33
def face_detector(img, size=0.5):
    
    # Convert image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return img, []
    
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))
    return img, roi

#################sending email and whatsapp message on Face recognition of narayan  ####################
# Open Webcam
cap = cv2.VideoCapture(0)
while True:

    ret, frame = cap.read()
    image, face = face_detector(frame)
    
    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Pass face to prediction model
        # "results" comprises of a tuple containing the label and the confidence value
        results = narayan_model.predict(face)
        if results[1] < 500:
            confidence = int( 100 * (1 - (results[1])/400) )
            display_string = str(confidence) + '% Confident it is Narayan'
            
        cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255,120,150), 2)
        
        if confidence > 85:
            cv2.putText(image, "Hey Narayan", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Face Recognition', image )
            #send the email 
            sendMail('abc@gmail.com','xyz@gmail.com',"Hi Narayan!!")
            #send the whatsapp message 
            sendWhatsAppMessage("1234567890","Hi!! How are You")
            break
         
        else:
            
            cv2.putText(image, "I dont know, how r u", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            cv2.imshow('Face Recognition', image )

    except:
        cv2.putText(image, "No Face Found", (220, 120) , cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.putText(image, "looking for face", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.imshow('Face Recognition', image )
        pass
        
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break     
cap.release()
cv2.destroyAllWindows()     


#################Launching ec2 instance and attaching ebs volume on face recognition of monil ####################
# Open Webcam
cap = cv2.VideoCapture(0)
while True:

    ret, frame = cap.read()
    image, face = face_detector(frame)
    
    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Pass face to prediction model
        # "results" comprises of a tuple containing the label and the confidence value
        results = monil_model.predict(face)
        if results[1] < 500:
            confidence = int( 100 * (1 - (results[1])/400) )
            display_string = str(confidence) + '% Confident it is Monil'
            
        cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255,120,150), 2)
        
        if confidence > 80:
            cv2.putText(image, "Hey Monil", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Face Recognition', image )
            print(ec2_ebs("keyaws","ap-south-1","ami-010aff33ed5991201","t2.micro","5"))
            print('Monil')
            break
         
        else:
            
            cv2.putText(image, "Face not recognized", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            cv2.imshow('Face Recognition', image )

    except:
        cv2.putText(image, "No Face Found", (220, 120) , cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.putText(image, "looking for face", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.imshow('Face Recognition', image )
        pass
        
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
cap.release()        
cv2.destroyAllWindows()     


