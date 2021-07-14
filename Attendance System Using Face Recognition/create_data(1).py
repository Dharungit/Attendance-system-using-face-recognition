# Creating database 
# It captures images and stores them in datasets 
# folder under the folder name of sub_data 
import cv2, sys, numpy, os 
import sqlite3
haar_file = 'haarcascade_frontalface_default.xml'

# All the faces data will be 
# present this folder 
datasets = 'datasets'


# These are sub data sets of folder, 
# for my faces I've used my name you can 
# change the label here 
sub_data = input('Enter your Name(Without Spaces): ')
Roll=input("Enter your Roll: ")
Class=input("Enter your Class: ")
# connection = sqlite3.connect("students_database.db")
# c = connection.cursor()
# c.execute("INSERT INTO student (Name, Rollnumber) VALUES(?, ?)",(sub_data,Roll))
# c1 = connection.cursor()
# c1.execute("INSERT INTO Attendance_Reports (Name, Rollnumber) VALUES(?, ?)",(sub_data,Roll))
# connection.commit()

Folder_name= sub_data + " "+ Class + " " + Roll
print(Folder_name)
path = os.path.join(datasets,Folder_name) # dataset/dharun.R A 19cs033
if not os.path.isdir(path):
	os.makedirs(path) 

# defining the size of images 
(width, height) = (130, 100)	 

#'0' is used for my webcam, 
# if you've any other camera 
# attached use '1' like this 
# os.system("espeak -ven-us+f3 -s170-ven-us+f3 -s170 'Please face the camera'")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+haar_file)

webcam = cv2.VideoCapture(0,cv2.CAP_DSHOW) 

# The program loops until it has 30 images of the face. 
count = 1
while count <= 500: 
	_, im = webcam.read()
	gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.2, 4)
	for (x, y, w, h) in faces: 
		cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2) 
		face = gray[y:y + h, x:x + w] 
		face_resize = cv2.resize(face, (width, height)) 
		cv2.imwrite('%s/% s.png' % (path, count), face_resize) 
	count += 1
	
	
	cv2.imshow('Training the Person', im) 
	key = cv2.waitKey(10) 

	if key == 27: 
		break

