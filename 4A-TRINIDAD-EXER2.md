## **Module 2.0: Feature Extraction and Object Detection**

**Instructions:**
Complete the following tasks using OpenCV and relevant libraries in Python (e.g., OpenCV, scikit-image).
Each task should be implemented in a separate code block. Submit your Python script or notebook with all cells executed, outputs displayed, and brief explanations describing your approach, observations, and results.

**SIFT Feature Extraction**

SIFT (Scale-Invariant Feature Transform) detects important points, called keypoints, in an image. These keypoints represent distinct and unique features, such as corners or edges, that can be identified even if the image is resized, rotated, or transformed. SIFT generates a descriptor for each keypoint, which helps in matching these points across images.

The code first loads the image, converts it to grayscale (because many feature detectors work better on grayscale images), and then uses the SIFT algorithm to detect keypoints. The keypoints are visualized on the image.

**Key Points:**

*   Keypoints are important image features.
*   Descriptors are used to describe and match these keypoints.

```python
!apt-get update
!apt-get install -y cmake build-essential pkg-config

!git clone https://github.com/opencv/opencv.git
!git clone https://github.com/opencv/opencv_contrib.git

!mkdir -p opencv/build
%cd opencv/build
!cmake -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D OPENCV_ENABLE_NONFREE=ON \
        -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
        -D BUILD_EXAMPLES=OFF ..
!make -j8
!make install
```

# **Task 1: SIFT Feature Extraction**

1. Load an image of your choice.
2. Use the SIFT (Scale-Invariant Feature Transform) algorithm to detect and compute keypoints and
descriptors in the image.
3. Visualize the keypoints on the image and display the result.
   
```python
import cv2
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('/content/image 1.jpeg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and descriptors
keypoints, descriptors = sift.detectAndCompute(gray_image, None)

# Draw keypoints on the image
image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)

# Display the image with keypoints
plt.imshow(cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB))
plt.title('SIFT Keypoints')
plt.show()
```
![image](https://github.com/user-attachments/assets/6d6f61ef-8c76-40d3-b468-b44e86f3f9f3)


**SURF Feature Extraction**

SURF (Speeded-Up Robust Features) is similar to SIFT but is optimized for speed. SURF focuses on finding features faster, making it useful for real-time applications. It also detects keypoints and generates
descriptors but uses a different mathematical approach to SIFT.

In the code, SURF is used to detect keypoints in a grayscale image, and the keypoints are visualized similarly to SIFT. The performance of SURF is usually faster than SIFT, but it might miss certain keypoints
that SIFT would detect.

**Key Points:**

*   SURF is faster than SIFT.
*   It can be a good choice for real-time applications.

# **Task 2: SURF Feature Extraction**

1. Load a different image (or the same one).
2. Apply the SURF (Speeded-Up Robust Features) algorithm to detect and compute keypoints and
descriptors.
3. Visualize and display the keypoints.

```python
# Load the image
image = cv2.imread('/content/image 1.jpeg')

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize SURF detector (You might need OpenCV-contrib for SURF)
surf = cv2.xfeatures2d.SURF_create()

# Detect keypoints and descriptors
keypoints, descriptors = surf.detectAndCompute(gray_image, None)

# Draw keypoints on the image
image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)

# Display the image with keypoints
plt.imshow(cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB))
plt.title('SURF Keypoints')
plt.show()
```
![image](https://github.com/user-attachments/assets/0f8af41b-ba59-42d8-8296-e1f82549d32b)


**ORB Feature Extraction**

ORB (Oriented FAST and Rotated BRIEF) is a feature detection algorithm that is both fast and computationally less expensive than SIFT and SURF. It is ideal for real-time applications, particularly in mobile devices. ORB combines two methods: FAST (Features from Accelerated Segment Test) to detect keypoints and BRIEF (Binary Robust Independent Elementary Features) to compute descriptors.

The code uses ORB to detect keypoints and display them on the image. Unlike SIFT and SURF, ORB is more focused on speed and efficiency, which makes it suitable for applications that need to process images
quickly.

**Key Points:**

*   ORB is a fast alternative to SIFT and SURF.
*   It’s suitable for real-time and resource-constrained environments.
*   
# **Task 3: ORB Feature Extraction (20 points)**

1. Apply the ORB (Oriented FAST and Rotated BRIEF) algorithm to detect keypoints and compute descriptors on another image.
2. Visualize and display the keypoints.

```python
import cv2
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('/content/image 1.jpeg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize ORB detector
orb = cv2.ORB_create()

# Detect keypoints and descriptors
keypoints, descriptors = orb.detectAndCompute(gray_image, None)

# Draw keypoints on the image
image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)

# Display the image with keypoints
plt.imshow(cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB))
plt.title('ORB Keypoints')
plt.show()
```
![image](https://github.com/user-attachments/assets/dfeebfc1-8298-4ddb-9647-c73e87e41416)


**Feature Matching using SIFT**

In this exercise, feature matching is used to find similar points between two images. After detecting keypoints using SIFT, the algorithm uses a Brute-Force Matcher to find matching keypoints between two
images. The matcher compares the descriptors of the keypoints and finds pairs that are similar.

In the code, we load two images, detect their keypoints and descriptors using SIFT, and then use the matcher to draw lines between matching keypoints. The lines show which points in the first image correspond to points in the second image.

**Key Points:**

*   Feature matching helps compare and find similarities between two images.
*   The Brute-Force Matcher finds the closest matching descriptors.

# **Task 4: Feature Matching**

1. Using the keypoints and descriptors obtained from the previous tasks (e.g., SIFT, SURF, or ORB), match the features between two different images using Brute-Force Matching or FLANN (Fast Library for Approximate Nearest Neighbors).
2. Display the matched keypoints on both images.

```python
import cv2
import matplotlib.pyplot as plt

# Load two images
image1 = cv2.imread('/content/image 1.jpeg', 0)
image2 = cv2.imread('/content/image 2.jpeg', 0)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Find keypoints and descriptors with SIFT
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

# Initialize the matcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Match descriptors
matches = bf.match(descriptors1, descriptors2)

# Sort matches by distance (best matches first)
matches = sorted(matches, key=lambda x: x.distance)

# Draw matches
image_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the matches
plt.imshow(image_matches)
plt.title('Feature Matching with SIFT')
plt.show()
```
![image](https://github.com/user-attachments/assets/9cfbc60e-d6b8-4070-9a97-85d5643bf851)

**Real-World Applications (Image Stitching using Homography)**

In this task, you use matched keypoints from two images to align or "stitch" them together. Homography is a mathematical transformation that maps points from one image to another, which is useful for aligning
images taken from different angles or perspectives. This process is used in image stitching (e.g., creating panoramas), where you align and merge images to form a larger one.

The code uses the keypoints matched between two images and calculates the homography matrix. This matrix is then used to warp one image to align it with the other.

**Key Points:**

*    Homography is used to align images.
*   This is useful in applications like panoramic image creation or object recognition.

# **Task 5:Applications of Feature Matching**
1. Apply feature matching to two images of the same scene taken from different angles or perspectives.
2. Use the matched features to align the images (e.g., using homography to warp one image onto another).

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load two images
image1 = cv2.imread('/content/image 2.jpeg')
image2 = cv2.imread('/content/image 1.jpeg')

# Convert to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Detect keypoints and descriptors using SIFT
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# Match features using BFMatcher
bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# Apply ratio test (Lowe's ratio test)
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Extract location of good matches
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Find homography matrix
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Warp one image to align with the other
h, w, _ = image1.shape
result = cv2.warpPerspective(image1, M, (w, h))

# Display the result
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title('Image Alignment using Homography')
plt.show()
```
![image](https://github.com/user-attachments/assets/c606e633-99a2-4acf-8f98-3485706360fa)


**Combining SIFT and ORB**

By combining two feature extraction methods (SIFT and ORB), you can take advantage of the strengths of both. For example, SIFT is more accurate, but ORB is faster. By detecting keypoints using both methods,
you can compare how they perform on different types of images and possibly combine their outputs for more robust feature detection and matching.

In the code, we extract keypoints from two images using both SIFT and ORB, and then you can use a matcher to compare and match the features detected by both methods.

**Key Points:**

*  Combining methods can improve performance in some applications.
*   SIFT is accurate, while ORB is fast, making them complementary in certain tasks.

# **Task 6: Combining Feature Extraction Methods (10 points)**
1. Combine multiple feature extraction methods (e.g., SIFT + ORB) to extract features and match them between two images.
2. Display the combined result.

```python
import cv2

image1 = cv2.imread('/content/image 1.jpeg', 0)
image2 = cv2.imread('/content/image 2.jpeg', 0)

sift = cv2.SIFT_create()
keypoints1_sift, descriptors1_sift = sift.detectAndCompute(image1, None)
keypoints2_sift, descriptors2_sift = sift.detectAndCompute(image2, None)

orb = cv2.ORB_create()
keypoints1_orb, descriptors1_orb = orb.detectAndCompute(image1, None)
keypoints2_orb, descriptors2_orb = orb.detectAndCompute(image2, None)
```
```python
bf_sift = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches_sift = bf_sift.match(descriptors1_sift, descriptors2_sift)
matches_sift = sorted(matches_sift, key=lambda x: x.distance)

bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches_orb = bf_orb.match(descriptors1_orb, descriptors2_orb)
matches_orb = sorted(matches_orb, key=lambda x: x.distance)

image_sift_matches = cv2.drawMatches(image1, keypoints1_sift, image2, keypoints2_sift, matches_sift[:30], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

image_orb_matches = cv2.drawMatches(image1, keypoints1_orb, image2, keypoints2_orb, matches_orb[:30], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
```

```python
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(image_sift_matches)
plt.title('SIFT Matches')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(image_orb_matches)
plt.title('ORB Matches')
plt.axis('off')

plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/7bba9382-de8e-49c9-a215-fc6713c41978)


# **Overall Understanding:**

*   SIFT is accurate for detecting and matching keypoints even in transformed images (scaled, rotated).
*   SURF is faster than SIFT but still effective for keypoint detection.
*   ORB is highly efficient and suited for real-time applications.
*   Feature Matching is essential in comparing different images to find common objects or align them.
*   Homography is used in aligning images, such as stitching images together to form a panorama.

These methods are foundational in computer vision tasks, such as object detection, image stitching, and image recognition



