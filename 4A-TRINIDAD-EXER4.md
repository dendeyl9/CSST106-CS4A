# **Exercise 1: Harris Corner Detection**

**Task:** Harris Corner Detection is a classic corner detection algorithm. Use the Harris Corner Detection algorithm to detect corners in an image.

* Load an image of your choice.
* Convert it to grayscale.
* Apply the Harris Corner Detection method to detect corners.
* Visualize the corners on the image and display the result.

**Key Points:**

* Harris Corner Detection is used to find corners, which are points of interest.
* It’s particularly useful for corner detection in images where object edges intersect.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('cylaaa.jpeg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray_image = np.float32(gray_image)
dst = cv2.cornerHarris(gray_image, blockSize=2, ksize=3, k=0.04)

dst = cv2.dilate(dst, None)
image[dst > 0.01 * dst.max()] = [0, 0, 255]
  
plt.figure(figsize=(4, 4))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Harris Corner Detection')
plt.axis('off')
plt.show()
```
![image](https://github.com/user-attachments/assets/7517e508-6051-44b6-b34c-0d13b1de2585)

**Explanation:** 
The Harris Corner Detection algorithm is successfully used to identify corners in an image in exercise 1. The image had to be loaded, converted to grayscale, and then corner points were identified using the Harris method. On the original image, the identified corners were highlighted in red to make them visible. This technique works well for finding intersections between edges, which is helpful for a number of image processing tasks like motion tracking and object recognition.


# **Exercise 2:** HOG (Histogram of Oriented Gradients) Feature Extraction

**Task:** The HOG descriptor is widely used for object detection, especially in human detection.

* Load an image of a person or any object.
* Convert the image to grayscale.
* Apply the HOG descriptor to extract features.
* Visualize the gradient orientations on the image.

**Key Points:**
* HOG focuses on the structure of objects through gradients.
* Useful for human detection and general object recognition.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure
  
image = cv2.imread('cylaaa.jpeg')
  
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
hog_features, hog_image = hog(
      gray_image,
      pixels_per_cell=(8, 8),
      cells_per_block=(2, 2),
      block_norm='L2-Hys',
      visualize=True,
      feature_vector=True
  )
  
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
  
plt.figure(figsize=(8, 4))
  
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')
  
plt.subplot(1, 2, 2)
plt.imshow(hog_image_rescaled, cmap='gray')
plt.title('HOG Visualization')
plt.axis('off')
  
plt.show()
```
![image](https://github.com/user-attachments/assets/bc836e2e-f99b-4b2f-84f6-cb129d9d9058)

**Explanation:**
The Histogram of Oriented Gradients (HOG) feature extraction method is used to look at the structure of an image in this task. The HOG descriptor captures the object's key shape and texture information by converting the image to grayscale and extracting gradient-based features. The visualization of the gradient orientations demonstrated how HOG emphasizes edges and contours. Because it offers reliable features that characterize an object's appearance and structure, this approach is especially helpful for tasks like general object recognition and human detection.

# **Exercise 3: FAST (Features from Accelerated Segment Test) Keypoint Detection**

**Task:** FAST is another keypoint detector known for its speed.

* Load an image.
* Convert the image to grayscale.
* Apply the FAST algorithm to detect keypoints.
* Visualize the keypoints on the image and display the result.

**Key Points:**
* FAST is designed to be computationally efficient and quick in detecting keypoints.
* It is often used in real-time applications like robotics and mobile vision.

```python
image = cv2.imread('cylaaa.jpeg')

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

fast = cv2.FastFeatureDetector_create()

keypoints = fast.detect(gray_image, None)

image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB))
plt.title('FAST Keypoints')
plt.axis('off')

plt.show()
```
![image](https://github.com/user-attachments/assets/abf23ef7-9947-410f-96f9-4ff8b839560d)

**Explanation:**
Keypoints in an image are found using the FAST (Features from Accelerated Segment Test) keypoint detection algorithm. FAST is used to quickly identify keypoints after the image has been converted to grayscale. By highlighting these key points on the picture, the areas of interest are quickly visible. The FAST algorithm's computational efficiency is its main strength, which makes it appropriate for real-time applications where speed is essential for feature detection in situations that change, such as robotics and mobile vision.

# **Exercise 4: Feature Matching using ORB and FLANN**

**Task:** Use ORB descriptors to find and match features between two images using FLANN-based matching.

* Load two images of your choice.
* Extract keypoints and descriptors using ORB.
* Match features between the two images using the FLANN matcher.
* Display the matched features.

**Key Points:**
* ORB is fast and efficient, making it suitable for resource-constrained environments.
* FLANN (Fast Library for Approximate Nearest Neighbors) speeds up the matching process, making it ideal for large datasets.

```python
image1 = cv2.imread('cylaaa.jpeg')
image2 = cv2.imread('face.jpeg')

height, width = image1.shape[:2]
image2 = cv2.resize(image2, (width, height))

gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create()

keypoints1, descriptors1 = orb.detectAndCompute(gray_image1, None)
keypoints2, descriptors2 = orb.detectAndCompute(gray_image2, None)

index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(descriptors1, descriptors2, k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.figure(figsize=(10, 4))
plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
plt.title('Feature Matching using ORB and FLANN with Resized Images')
plt.axis('off')
plt.show()
```
![image](https://github.com/user-attachments/assets/76d5f931-f8e6-4e5c-86ee-999fc3c30f07)

**Explanation:**
FLANN matchers and ORB descriptors are used to match features between images. Images are loaded and converted to grayscale at the beginning of the process. ORB can be used for real-time settings since it can identify keypoints and extract descriptors. FLANN is perfect for large datasets because it matches features quickly. High-quality correspondences are retained after bad matches are eliminated using Lowe's ratio test. The visual representation of the matched keypoints highlights similarities and is helpful for motion tracking, object recognition, and image stitching.

# **Exercise 5: Image Segmentation using Watershed Algorithm**

**Task:** The Watershed algorithm segments an image into distinct regions.

* Load an image.
* Apply a threshold to convert the image to binary.
* Apply the Watershed algorithm to segment the image into regions.
* Visualize and display the segmented regions.

**Key Points:**
* Image segmentation is crucial for object detection and recognition.
* The Watershed algorithm is especially useful for separating overlapping objects.

```python
image_path = 'cylaaa.jpeg' 
original_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(thresh, kernel, iterations=3)
dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(dilated, sure_fg)

_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0
segmented = cv2.watershed(original_image, markers) 

original_image[segmented == -1] = [255, 0, 0] 

plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(thresh, cmap='gray')
plt.title('Binary Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(original_image)
plt.title('Segmented Image by using Watershed Algorithm')
plt.axis('off')

plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/921cbaba-36c9-4615-b3e8-ba10df931ee1)

**Explanation:**
The use of the Watershed algorithm for image segmentation, a technique used for separating overlapping objects in images. This technique is crucial in object detection and recognition, as it allows for the isolation of distinct regions of interest. The process involves loading the image, converting it to grayscale, applying Gaussian blur, and creating a binary representation. The Watershed algorithm then delineates distinct regions based on pixel intensity gradients, making it useful in situations where objects overlap or are in close proximity. The segmented output highlights the algorithm's effectiveness in image analysis tasks.





