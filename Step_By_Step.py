import numpy as np 
import cv2
import dlib

### Step 1: Get a hat (.png) and convert it

hat_img = cv2.imread("hat2.png",-1)  # read in the original hat image
r,g,b,a = cv2.split(hat_img)  # alpha channel as mask
rgb_hat = cv2.merge((r,g,b))
cv2.imwrite("hat_alpha.jpg",a)

img = cv2.imread("test2.JPG")

### Step 2: detect the face -- using dlib dectector

# shape_predictor_5_face_landmarks.dat
predictor_path = "shape_predictor_5_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)  

# dlib frontal face detector
detector = dlib.get_frontal_face_detector()

# face detector
dets = detector(img, 1)

# If dectect the human face

if len(dets)>0:
	for d in dets:
		x,y,w,h = d.left(),d.top(), d.right()-d.left(), d.bottom()-d.top()
		# x,y,w,h = faceRect
		cv2.rectangle(img,(x,y), (x+w,y+h),(255,0,0), 2,8,0)
		shape = predictor(img, d)

		# five key points
		for point in shape.parts():
			cv2.circle(img,(point.x, point.y),3,color=(0,255,0))


		#cv2.imshow("image", img)
		#cv2.waitKey()

### Step 3: Adjust the position of hat

# pick the eye angles points 
point1 = shape.part(0)
point2 = shape.part(2)

# get the eye center
eyes_center = ((point1.x+point2.x)//2,(point1.y+point2.y)//2)

# adjust the size of hat based on the size of face
factor = 1.5
resized_hat_h = int(round(rgb_hat.shape[0]*w/rgb_hat.shape[1]*factor))
resized_hat_w = int(round(rgb_hat.shape[1]*w/rgb_hat.shape[1]*factor))

if resized_hat_h > y:
	resized_hat_h = y-1

resized_hat = cv2.resize(rgb_hat,(resized_hat_w,resized_hat_h))

### Step 4: Get the hat region

# use alpha channel as mask
mask = cv2.resize(a,(resized_hat_w,resized_hat_h))
mask_inv =  cv2.bitwise_not(mask)

# the gap between hat and face

dh = 0
dw = 0
# original ROI 
bg_roi = img[y+dh-resized_hat_h:y+dh,(eyes_center[0]-resized_hat_w//3):(eyes_center[0]+resized_hat_w//3*2)]
# ROI hat region
bg_roi = bg_roi.astype(float)
mask_inv = cv2.merge((mask_inv,mask_inv,mask_inv))
alpha = mask_inv.astype(float)/255
# ensure the same size
alpha = cv2.resize(alpha,(bg_roi.shape[1],bg_roi.shape[0]))
print("alpha size: ",alpha.shape)
print("bg_roi size: ",bg_roi.shape)
bg = cv2.multiply(alpha, bg_roi)
bg = bg.astype('uint8')


hat = cv2.bitwise_and(resized_hat,resized_hat,mask = mask)
cv2.imwrite("hat.jpg",hat)

### Step 5: Add the Hat

hat = cv2.resize(hat,(bg_roi.shape[1],bg_roi.shape[0]))

add_hat = cv2.add(bg,hat)


img[y+dh-resized_hat_h:y+dh,(eyes_center[0]-resized_hat_w//3):(eyes_center[0]+resized_hat_w//3*2)] = add_hat
cv2.imshow("img",img )
cv2.waitKey(0)  



