# import numpy as np
# import cv2

# cap = cv2.VideoCapture('Megamind.avi')

# while(cap.isOpened()):
#     ret, frame = cap.read()

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     cv2.imshow('frame',gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


# import numpy as np
# import cv2

# cap = cv2.VideoCapture(0)
 
# while(True):
#     ret, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
#     cv2.imshow('frame',gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


# OpenCV Python program to detect cars in video frame
#import libraries of python OpenCV 
import cv2
import pdb
 
# capture frames from a video
cap = cv2.VideoCapture('crash15.mp4')
 
# Trained XML classifiers describes some features of some object we want to detect
car_cascade = cv2.CascadeClassifier('cars.xml')
 
# loop runs if capturing has been initialized.
while True:

	# reads frames from a video
	ret, frames = cap.read()
	 
	# convert to gray scale of each frames
	gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
	 

	# Detects cars of different sizes in the input image
	cars = car_cascade.detectMultiScale(gray, 1.1, 1)
	 
	# To draw a rectangle in each cars
	for (x,y,w,h) in cars:
	    cv2.rectangle(frames,(x,y),(x+w,y+h),(0,0,255),2)

	# Display frames in a window 
	cv2.imshow('video2', frames)
	 
	# Wait for Esc key to stop
	if cv2.waitKey(33) == 27:
	    break
 
# De-allocate any associated memory usage
cv2.destroyAllWindows()


# import numpy as np
# import cv2
# import pdb
# import matplotlib.pyplot as plt
# from scipy.ndimage.filters import gaussian_filter


# cap = cv2.VideoCapture('crash_tbone.avi')

# # params for ShiTomasi corner detection
# feature_params = dict( maxCorners = 100,
#                        qualityLevel = 0.3,
#                        minDistance = 7,
#                        blockSize = 7 )

# # Parameters for lucas kanade optical flow
# lk_params = dict( winSize  = (20,20),
#                   maxLevel = 2,
#                   criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

# # Create some random colors
# color = np.random.randint(0,255,(100,3))

# # Take first frame and find corners in it
# ret, old_frame = cap.read()
# old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
# p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# # Create a mask image for drawing purposes
# mask = np.zeros_like(old_frame)

# pss = []

# while(1):
#     ret,frame = cap.read()
#     frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # calculate optical flow
#     p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

#     # pss.append(p1[18][0])
#     pss.append(p1[0][0])

#     # Select good points
#     good_new = p1[st==1]
#     good_old = p0[st==1]


#     # draw the tracks
#     for i,(new,old) in enumerate(zip(good_new,good_old)):
#     	#if i == 0:
# 	        a,b = new.ravel()
# 	        c,d = old.ravel()
# 	        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
# 	        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
#     img = cv2.add(frame,mask)

#     cv2.imshow('frame',img)
#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break

#     # Now update the previous frame and previous points
#     old_gray = frame_gray.copy()
#     p0 = good_new.reshape(-1,1,2)

# cv2.destroyAllWindows()
# cap.release()




# t = [i for i in range(len(pss))]
# x = []
# y = []
# for xi,yi in pss:
# 	x.append(xi)
# 	y.append(yi)

# v = []
# dt = 0.5
# for i in range(1,len(pss)):
# 	x0 = np.array(pss[i-1])
# 	x1 = np.array(pss[i])
# 	dist = np.linalg.norm(x0-x1)
# 	v.append(dist/dt)


# v = gaussian_filter(v, sigma=1)

# a = []
# for i in range(1,len(v)):
# 	v0 = np.array(v[i-1])
# 	v1 = np.array(v[i])
# 	dv = np.linalg.norm(v0-v1)
# 	a.append(dv/dt)


# plt.figure()
# plt.subplot(211)
# plt.plot(t,x)
# plt.plot(t,y)
# plt.grid("on")
# plt.subplot(212)
# plt.plot(t[1:],v)
# plt.plot(t[2:],a)
# plt.grid("on")
# plt.show()
# plt.close()





# import cv2
# import numpy as np
# cap = cv2.VideoCapture("crashes.avi")

# ret, frame1 = cap.read()
# prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
# hsv = np.zeros_like(frame1)
# hsv[...,1] = 255

# while(1):
#     ret, frame2 = cap.read()
#     next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

#     flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

#     mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
#     hsv[...,0] = ang*180/np.pi/2
#     hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
#     rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

#     cv2.imshow('frame2',rgb)
#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break
#     elif k == ord('s'):
#         cv2.imwrite('opticalfb.png',frame2)
#         cv2.imwrite('opticalhsv.png',rgb)
#     prvs = next

# cap.release()
# cv2.destroyAllWindows()

