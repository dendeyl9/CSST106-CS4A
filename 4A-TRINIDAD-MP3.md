 # **Module 2.0: Feature Extraction and Object Detection**
**Machine Problem No. 3: Feature Extraction and Object Detection**

**Objective:**
The objective of this machine problem is to implement and compare the three feature extraction methods (SIFT, SURF, and ORB) in a single task. 
You will use these methods for feature matching between two images, then perform image alignment using homography to warp one image onto the other.

**Problem Description:**

You are tasked with loading two images and performing the following steps:
1. Extract keypoints and descriptors from both images using SIFT, SURF, and ORB.
2. Perform feature matching between the two images using both Brute-Force Matcher and FLANN Matcher.
3. Use the matched keypoints to calculate a homography matrix and align the two images.
4. Compare the performance of SIFT, SURF, and ORB in terms of feature matching accuracy and speed.

You will submit your code, processed images, and a short report comparing the results.

**Task Breakdown:**

**Step 1: Load Images**

*   Load two images of your choice that depict the same scene or object but from different angles.

**Step 2: Extract Keypoints and Descriptors Using SIFT, SURF, and ORB (30 points)**

*   Apply the SIFT algorithm to detect keypoints and compute descriptors for both images.
*   Apply the SURF algorithm to do the same.
*   Finally, apply ORB to extract keypoints and descriptors.

 ```python
import cv2
import matplotlib.pyplot as plt

# Load two images that depict the same scene or object from different angles
image1 = cv2.imread('/content/image 1.jpeg', 0)  # Load the first image in grayscale
src = cv2.imread('/content/image 2.jpeg', 0)  # Load the second image in grayscale
image2 = cv2.rotate(src, cv2.ROTATE_90_CLOCKWISE)
# Check if images are loaded correctly

plt.figure(figsize=(5, 5))

# Show first image
plt.subplot(1, 2, 1)
plt.imshow(image1, cmap='gray')
plt.title('Image 1')
plt.axis('off')

# Show second image
plt.subplot(1, 2, 2)
plt.imshow(image2, cmap='gray')
plt.title('Image 2')
plt.axis('off')

plt.tight_layout()
plt.show()
 ```
![image](https://github.com/user-attachments/assets/4f51ddb1-1941-461b-8c81-bf73d43d016e)

```python
sift = cv2.SIFT_create()
keypoints1_sift, descriptors1_sift = sift.detectAndCompute(image1, None)
keypoints2_sift, descriptors2_sift = sift.detectAndCompute(image2, None)

surf = cv2.xfeatures2d.SURF_create()
keypoints1_surf, descriptors1_surf = surf.detectAndCompute(image1, None)
keypoints2_surf, descriptors2_surf = surf.detectAndCompute(image2, None)

orb = cv2.ORB_create()
keypoints1_orb, descriptors1_orb = orb.detectAndCompute(image1, None)
keypoints2_orb, descriptors2_orb = orb.detectAndCompute(image2, None)

image1_sift_keypoints = cv2.drawKeypoints(image1, keypoints1_sift, None, color=(255, 0, 0))
image2_sift_keypoints = cv2.drawKeypoints(image2, keypoints2_sift, None, color=(255, 0, 0))

image1_surf_keypoints = cv2.drawKeypoints(image1, keypoints1_surf, None, color=(0, 255, 0))
image2_surf_keypoints = cv2.drawKeypoints(image2, keypoints2_surf, None, color=(0, 255, 0))

image1_orb_keypoints = cv2.drawKeypoints(image1, keypoints1_orb, None, color=(0, 0, 255))
image2_orb_keypoints = cv2.drawKeypoints(image2, keypoints2_orb, None, color=(0, 0, 255))

plt.figure(figsize=(10, 8))

plt.subplot(3, 2, 1)
plt.imshow(cv2.cvtColor(image1_sift_keypoints, cv2.COLOR_BGR2RGB))
plt.title('SIFT Keypoints (Image 1)')
plt.axis('off')

plt.subplot(3, 2, 2)
plt.imshow(cv2.cvtColor(image2_sift_keypoints, cv2.COLOR_BGR2RGB))
plt.title('SIFT Keypoints (Image 2)')
plt.axis('off')

plt.subplot(3, 2, 3)
plt.imshow(cv2.cvtColor(image1_surf_keypoints, cv2.COLOR_BGR2RGB))
plt.title('SURF Keypoints (Image 1)')
plt.axis('off')

plt.subplot(3, 2, 4)
plt.imshow(cv2.cvtColor(image2_surf_keypoints, cv2.COLOR_BGR2RGB))
plt.title('SURF Keypoints (Image 2)')
plt.axis('off')

plt.subplot(3, 2, 5)
plt.imshow(cv2.cvtColor(image1_orb_keypoints, cv2.COLOR_BGR2RGB))
plt.title('ORB Keypoints (Image 1)')
plt.axis('off')

plt.subplot(3, 2, 6)
plt.imshow(cv2.cvtColor(image2_orb_keypoints, cv2.COLOR_BGR2RGB))
plt.title('ORB Keypoints (Image 2)')
plt.axis('off')

plt.tight_layout()
plt.show()
```

![image](https://github.com/user-attachments/assets/6a38b97f-ab12-4e97-8a54-487330b6b679)

