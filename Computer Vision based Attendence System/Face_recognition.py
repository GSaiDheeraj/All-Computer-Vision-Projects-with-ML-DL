import cv2
import face_recognition

#Step 1 Importing our test image and traing image
imgElon = face_recognition.load_image_file('Images/ElonMusk.jpg')
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('Images/billagtes.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

# Step 2finding faces and find encodings
faceLoc = face_recognition.face_locations(imgElon)[0] #We will send first element of the image
encodeElon = face_recognition.face_encodings(imgElon)[0] # encode the image
#locate the faces detected and we hav to give locations of the faces(top right bottom, left, right)
cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 0), 2)

#Find the encodings of the test image similar to above process
faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (0, 0, 255), 2)

#Step3 compare encodings
# we use  linear SVM
results = face_recognition.compare_faces([encodeElon], encodeTest) # o/p:true
# because it find similar encodings
#Step4: finding distances
#there may be similar encodings for some images so we find the distance
faceDis = face_recognition.face_distance([encodeElon], encodeTest)
print(results, faceDis) #lower the distance, better the match
#to visualize the results we use put text
#Step5: Visulaize the finding
cv2.putText(imgTest, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('Elon Musk', imgElon)
cv2.imshow('Elon Test', imgTest)
cv2.waitKey(0)