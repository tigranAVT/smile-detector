import numpy as np
import cv2

face_clf = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_clf = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_clf = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml') 

def detect(gray, frame): 
	faces = face_clf.detectMultiScale(gray, 1.3, 5) 
	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 130, 0), 2)
		roi_gray = gray[y:y + h, x:x + w] 
		roi_color = frame[y:y + h, x:x + w] 
		eyes = eye_clf.detectMultiScale(roi_gray, 1.2, 24)
		for (ex, ey, ew, eh) in eyes: 
			cv2.rectangle(roi_color, (ex, ey), ((ex + ew), (ey + eh)), (0, 0, 255), 2)
			smiles = smile_clf.detectMultiScale(roi_gray, 1.7, 62) 
			for (sx, sy, sw, sh) in smiles:
				smiles_roi = roi_color[sy: sy + sh, sx:sx + sw] 
				cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 180), 2)
				if len(eyes) == 2:
					cv2.imwrite("./img/output.jpg", frame)

	return frame 

video_capture = cv2.VideoCapture(0) 

while True: 
	_, frame = video_capture.read() 
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
	canvas = detect(gray, frame) 				 
	cv2.imshow('Webcam', canvas) 
	if cv2.waitKey(1) & 0xff == ord('q'):			 
		break

video_capture.release()								 
cv2.destroyAllWindows() 
