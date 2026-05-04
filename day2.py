import numpy as np
import cv2
import matplotlib.pyplot as plt


#image = np.array([[100, 150, 200, 250 ],
#              [50,75, 125, 175 ],
#               [30, 60, 90, 120],
#                [10, 20, 30, 40]
#       ], dtype=np.uint8)

image =cv2.imread(r"C:\Users\haroo\Downloads\cat.jpg")

print('shape: ', image.shape)
print("Max pixel: ",np.max(image))
print("Mean (average brightness):", image.mean())

image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)



crop = image[0:2, 0:8]
print("Cropped image: \n", crop)



Normalized_image =image /255.0
print ("Normalized Max :", Normalized_image.max())

flipped =np.fliplr(image)
print("Flipped :\n", flipped)

plt.figure(figsize=(6,6))
plt.subplot(1,2,1)
plt.imshow(image)
plt.title("Original Image")


plt.subplot(1,2,2)
plt.imshow(flipped, cmap='gray')
plt.title("flipped Image")

plt.axis('off')
plt.show()

# detecting edges using Canny edge detection
edge = cv2.Canny(image,threshold1=120,threshold2=220)

plt.figure(figsize=(6,6))
plt.subplot(1,2,1)
plt.imshow(image)
plt.title("Original Image")


plt.subplot(1,2,2)
plt.imshow(edge, cmap='gray')
plt.title("edge Image")

plt.axis('off')
plt.show()



#pre processing the image by resizing it to a fixed size

gray =cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# reszize to standard CNN size
img=cv2.resize(grey,(224,224))

#normailizing image from 0-1
img=img.astype(np.float32) /255.0

