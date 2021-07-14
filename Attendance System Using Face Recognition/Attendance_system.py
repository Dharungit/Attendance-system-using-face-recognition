import cv2, numpy, os 
import mysql
import mysql.connector


haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'

mydb= mysql.connector.connect(host="localhost", user='root', passwd="#@msql@FEW#", database='Students')
cur= mydb.cursor()

def markAttendance(name,rollnumber,student_class,Attendance):
    try:
        #enter Attendence into database
        # print(student_class)
        if student_class=="A":
            sqlform='insert into class_a(NAME,ROLLNUMBER,Attendance,Class) values(%s,%s,%s,%s)'
        elif student_class=="B":
            sqlform='insert into class_b(NAME,ROLLNUMBER,Attendance,Class) values(%s,%s,%s,%s)'
        elif student_class=="C":
            sqlform='insert into class_c(NAME,ROLLNUMBER,Attendance,Class) values(%s,%s,%s,%s)'
        
        st=[(name,rollnumber,Attendance,student_class)]

        cur.executemany(sqlform,st)
        mydb.commit()
    except:
        pass


#Training
(images, lables, names, Class,rollnum, id) = ([], [], {},{},{},0) 
for (subdirs, dirs, files) in os.walk(datasets): 
	# print("images-->",images)
	# print("Subdirs-->",dirs)
	# print("labels-->",lables)
	for subdir in dirs:  
		# print("Subdirs-->",subdir)
		names[id],Class[id],rollnum[id] = subdir.split()
		subjectpath = os.path.join(datasets, subdir)  #dharun.R A 19cs033
		for filename in os.listdir(subjectpath):
			# print("filename-->",filename) 
			path = subjectpath + '/' + filename  #dataset/dharun.R A 19cs033/3.png
			lable = id
			images.append(cv2.imread(path, 0)) 
			lables.append(int(lable)) 
		id += 1

(width, height) = (130, 100) 
# print(names)
# print(Class)
# print(rollnum)

# Create a Numpy array from the two lists above 
(images, lables) = [numpy.array(lis) for lis in [images, lables]] 

# OpenCV trains a model from the images 
# NOTE FOR OpenCV2: remove '.face' 
model = cv2.face.LBPHFaceRecognizer_create() 
model.train(images, lables) 

# Part 2: Use fisherRecognizer on camera stream 

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+haar_file) 
webcam = cv2.VideoCapture(0,cv2.CAP_DSHOW) 
nm,cls,rol=[],[],[]
while True: 
	(_, im) = webcam.read() 
	gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 
	faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
	for (x, y, w, h) in faces: 
		#cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2) 
		face = gray[y:y + h, x:x + w] 
		face_resize = cv2.resize(face, (width, height)) 
		# Try to recognize the face 
		prediction = model.predict(face_resize) 
		
		if prediction[1]<=100: 
			cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 3) 

			cv2.putText(im, '% s' %(names[prediction[0]]),(x-10, y-10),cv2.FONT_HERSHEY_TRIPLEX,1,(0, 255, 0)) 
			name=names[prediction[0]]
			std_class=Class[prediction[0]]
			std_rollnum=rollnum[prediction[0]]
			markAttendance(name,std_rollnum,std_class,"Present")

		else: 
			cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 225), 3) 
			
			cv2.putText(im, 'Unknown Person',(x-10, y-10), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0)) 
	
	cv2.imshow('Students Detection', im) 
	
	
	key = cv2.waitKey(27) 
	if key == 27: 
		break
