import numpy as np 
from imutils.video.filevideostream import FileVideoStream
import argparse
import time
import math
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the image")
ap.add_argument("-v", "--video", required=True, help="path to the video")
args = vars(ap.parse_args())

SCALE_PERCENT = 80 

temp_x = 0
temp_y = 0

offset = 2


face_cascade = cv2.CascadeClassifier('/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/usr/local/share/opencv4/haarcascades/haarcascade_eye.xml')

specs_image = cv2.imread(args['image'], -1)

width = int(specs_image.shape[1] * SCALE_PERCENT / 100)
height = int(specs_image.shape[0] * SCALE_PERCENT / 100)
dim = (width, height)

specs_image = cv2.resize(specs_image, dim)

orig_mask = specs_image[:, :, 3]
orig_mask_inv = cv2.bitwise_not(orig_mask)

specs_image = cv2.cvtColor(specs_image, cv2.COLOR_RGB2BGR)

specs_image = specs_image[:, :, 0:3]
specs_image_height, specs_image_width = specs_image.shape[:2]

# def angle(coordinate1, coordinate2):
# 	x1, y1 = coordinate1[0], coordinate1[1]
# 	x2, y2 = coordinate2[0], coordinate2[1]

# 	deltaX = x2-x1
# 	deltaY = y2-y1

# 	degrees_temp = math.atan2(deltaX, deltaY)/math.pi*180

# 	if degrees_temp < 0:

#         degrees = 360 + degrees_temp

#     else:

#         degrees = degrees_temp

# 	return degrees

fvs = FileVideoStream(args["video"]).start()
time.sleep(2)

# print(vs.stream.get(cv2.CAP_PROP_FPS))

while True:
	frame = fvs.read()

	if frame is None:
		break

	width = int(frame.shape[1] * SCALE_PERCENT / 100)
	height = int(frame.shape[0] * SCALE_PERCENT / 100)
	dim = (width, height)
	print(dim)

	frame = cv2.resize(frame, dim)

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(15, 15))

	if len(faces) > 0:
		for (x, y, w, h) in faces:
			if w*h > 10000:
				# cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
				face_roi_gray = gray[y:y+h, x:x+w]
				face_roi_frame = frame[y:y+h, x:x+w]

				eyes = eye_cascade.detectMultiScale(face_roi_gray)

				eyes = sorted(eyes, key=lambda x: x[0])


				if len(eyes) >= 2:
					for eX, eY, eW, eH in eyes:
						if abs(temp_x-eX) > 2 and abs(temp_y-eY) > 2:
	
							temp_x, temp_y = eX, eY
							temp_w, twmp_h = eW, eH

							# cv2.rectangle(face_roi_frame, (eX, eY), (eX+eW, eY+eH), (0, 0, 255), 2)
							specs_width = int(3.1 * eW)
							specs_height = int((specs_width / specs_image_width) * specs_image_height)

							if face_roi_frame.shape[1]-eX < specs_width:
								specs_width = face_roi_frame.shape[1]-eX

							if face_roi_frame.shape[0]-eY < specs_height:
								specs_height = face_roi_frame.shape[0]-eY

							spectacle = cv2.resize(specs_image, (specs_width, specs_height))
							mask = cv2.resize(orig_mask, (specs_width, specs_height))
							mask_inv = cv2.resize(orig_mask_inv, (specs_width, specs_height))

							eyes_roi = face_roi_frame[eY+offset:eY+specs_height+offset, eX-offset:eX+specs_width-offset]

							eyes_roi_bg = cv2.bitwise_and(eyes_roi, eyes_roi, mask=mask_inv)
							eyes_roi_fg = cv2.bitwise_and(spectacle, spectacle, mask=mask)

							stiched_roi = cv2.add(eyes_roi_bg, eyes_roi_fg)

							face_roi_frame[eY+offset:eY+specs_height+offset, eX-offset:eX+specs_width-offset] = stiched_roi
							break

						else:

							# cv2.rectangle(face_roi_frame, (temp_x, temp_x), (temp_x+eW, temp_y+eH), (0, 0, 255), 2)

							specs_width = int(3.1 * temp_w)
							specs_height = int((specs_width / specs_image_width) * specs_image_height)

							if face_roi_frame.shape[1]-temp_x < specs_width:
								specs_width = face_roi_frame.shape[1]-temp_x

							if face_roi_frame.shape[0]-temp_y < specs_height:
								specs_height = face_roi_frame.shape[0]-temp_y

							spectacle = cv2.resize(specs_image, (specs_width, specs_height))
							mask = cv2.resize(orig_mask, (specs_width, specs_height))
							mask_inv = cv2.resize(orig_mask_inv, (specs_width, specs_height))

							eyes_roi = face_roi_frame[temp_y+offset:temp_y+specs_height+offset, temp_x-offset:temp_x+specs_width-offset]

							eyes_roi_bg = cv2.bitwise_and(eyes_roi, eyes_roi, mask=mask_inv)
							eyes_roi_fg = cv2.bitwise_and(spectacle, spectacle, mask=mask)

							stiched_roi = cv2.add(eyes_roi_bg, eyes_roi_fg)

							face_roi_frame[temp_y+offset:temp_y+specs_height+offset, temp_x-offset:temp_x+specs_width-offset] = stiched_roi

							break


	cv2.imshow("output", frame)

	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

fvs.stop()

