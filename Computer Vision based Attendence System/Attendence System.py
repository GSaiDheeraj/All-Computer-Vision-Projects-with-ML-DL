import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

#Step6: Creating the list that can find the path of images
path = 'Images'
images = [] #Images list
classNames = [] #Name of the images
myList = os.listdir(path)
print(myList) #to see the Images in the folder

#Step7: Looping through the images folder
for cl in myList: #cl is name of the image
    curImg = cv2.imread(f'{path}/{cl}') #reading eevery image in the list
    images.append(curImg) #append the image into empty list
    classNames.append(os.path.splitext(cl)[0]) #we need only billgates no need of extension like .jpg
print(classNames)

#Step8: Finding encoding automatically from the images
def findEncodings(images): #we require list of images
    encodeList = []
    for img in images: #loop through images
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #converting imaegs
        encode = face_recognition.face_encodings(img)[0] #find the encodings
        encodeList.append(encode) #append the enocdings list to the empty list
    return encodeList

#step12: Marking attendence in the csv file
def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',') # we split line based on comma
            nameList.append(entry[0])# our entry will be two values and we need only name
        if name not in nameList: # if there is no name we will just write the time.
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

    #### FOR CAPTURING SCREEN RATHER THAN WEBCAM
    # def captureScreen(bbox=(300,300,690+300,530+300)):
    #     capScr = np.array(ImageGrab.grab(bbox))
    #     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
    #     return capScr

encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0) # webcam access
#step9: Aceessing web cam
while True: # while loop to get each frame
    success, img = cap.read() #reading of images
# img = captureScreen()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25) #resizing the image into quater of its size
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB) # converting into RGB image
    facesCurFrame = face_recognition.face_locations(imgS) #find the locations in the frame
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)#find encodings
    #Step10: Iterate through all faces and compare with encodings we found
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace) # we find matchings
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace) # finding the face distances

        matchIndex = np.argmin(faceDis) #to find the lowest face distance
        #STEP11: IF we the mataches found we just put a bounding box and write name
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            #keep bounding boxes for the face
            y1, x2, y2, x1 = faceLoc # face locations
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4 # as our image is scaled, inorder to perform original image we need to multiply with 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2) # keeping rectangle
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED) #place for name
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow('Attended Person', img)
    cv2.waitKey(1)
