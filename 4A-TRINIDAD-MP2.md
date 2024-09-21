# PECEPTION AND COMPUTER VISION (CSST 106)

**Machine Problem No. 2: Applying Image Processing Techniques**

**Objective:** Understand and apply various image processing techniques, including image transformations and filtering, using tools like OpenCV. 
Gain hands-on experience in implementing these techniques and solving common image processing tasks.

**Instructions:**

**Research and Comprehend:**

 **Lecture on Image Processing Concepts:**

  *   Attend the lecture on image processing to gain a thorough understanding of the core concepts, including image transformations like scaling and rotation, as well as filtering techniques such as blurring and edge detection

# Hands-On Exploration:

* **Lab Session 1: Image Transformations**
  * **Scaling and Rotation:** Learn how to apply scaling and rotation transformations to images
using OpenCV.
  * **Implementation:** Practice these transformations on sample images provided in the lab.

 ```python
  !pip install opencv-python-headless
```

```python
from google.colab import files
from io import BytesIO
from PIL import Image

# Upload an image
uploaded = files.upload()

# Convert to OpenCV format
image_path = next(iter(uploaded))
# Get the image file name
image = Image.open(BytesIO(uploaded[image_path]))
image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
display_image(image, "Original Image")
```
![image](https://github.com/user-attachments/assets/cb65bf48-8326-462f-883b-cc72d5eca8e4)

```python
def scale_image(img, scale_factor):
  height, width = img.shape[:2]
  scaled_img = cv2.resize(img,
(int(width * scale_factor), int(height * scale_factor)), interpolation=cv2.INTER_LINEAR)
  return scaled_img

def rotate_image(img, angle):
  height, width = img.shape[:2]
  center = (width // 2, height // 2)
  matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
  rotated_img = cv2.warpAffine(img, matrix, (width, height))
  return rotated_img

scaled_image = scale_image(image, 0.5)
display_image(scaled_image, "Scaled Image (50%)")

rotated_image = rotate_image(image, 45)
display_image(rotated_image, "Rotated Image (45°)")
```
![image](https://github.com/user-attachments/assets/fdc4e225-137b-4d28-b697-689a63b4448b)

# **Lab Session 2: Filtering Techniques**

**Blurring and Edge Detection:**
Explore how to apply blurring filters and edge detection algorithms to images using OpenCV.
Implementation: Apply these filters to sample images to understand their effects.

```python
gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)
median_blur = cv2.medianBlur(image, 5)
bilateral_filter = cv2.bilateralFilter(image, 9, 75, 75)

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2RGB))
plt.title("Gaussian Blur")
plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(median_blur, cv2.COLOR_BGR2RGB))
plt.title("Median Blur")
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(bilateral_filter, cv2.COLOR_BGR2RGB))
plt.title("Bilateral Filter")

plt.show()
```

![image](https://github.com/user-attachments/assets/4f7bb71f-aa6b-49fe-b315-2864f76d640e)

```python
# Canny Edge Detection
edges = cv2.Canny(image, 100, 200)
display_image(edges, "Canny Edge Detection (100, 200)")
```

![image](https://github.com/user-attachments/assets/3502e98b-7a69-4a45-9902-59a4e08333b1)

```python
def sobel_edge_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel_combined = cv2.magnitude(sobelx, sobely)
    return sobel_combined

sobel_edges = sobel_edge_detection(image)
plt.imshow(sobel_edges, cmap='gray')
plt.title("Sobel Edge Detection")
plt.axis('off')
plt.show()
```

![image](https://github.com/user-attachments/assets/e2b2886a-b496-4198-aa22-36c5f62245eb)

```python
# Prewitt Edge Detection
def prewitt_edge_detection(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Prewitt operator kernels for x and y directions
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=int)
    kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)

    # Applying the Prewitt operator
    prewittx = cv2.filter2D(gray, cv2.CV_64F, kernelx)
    prewitty = cv2.filter2D(gray, cv2.CV_64F, kernely)

    # Combine the x and y gradients by converting to floating point
    prewitt_combined = cv2.magnitude(prewittx, prewitty)

    return prewitt_combined

# Apply Prewitt edge detection to the uploaded image
prewitt_edges = prewitt_edge_detection(image)
plt.imshow(prewitt_edges, cmap='gray')
plt.title("Prewitt Edge Detection")
plt.axis('off')
plt.show()
```
![image](https://github.com/user-attachments/assets/eb790ca0-646f-49a9-8149-62d077e89b39)

```python
# Laplacian Edge Detection
def laplacian_edge_detection(img):
  # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # Apply Laplacian operator
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    return laplacian

# Apply Laplacian edge detection to the uploaded image
laplacian_edges = laplacian_edge_detection(image)
plt.imshow(laplacian_edges, cmap='gray')
plt.title("Laplacian Edge Detection")
plt.axis('off')
plt.show()
```
![image](https://github.com/user-attachments/assets/bd3dafec-de5b-4ada-8624-1879f5f06e1c)

# Problem-Solving Session:
*  Common Image Processing Tasks:
*  Engage in a problem-solving session focused on common challenges encountered in image processing tasks.

**Scenario-Based Problems:** 
Solve scenarios where you must choose and apply appropriate image processing techniques.

**Scenario:**
Object Detection in a Noisy Image

**Problem:**
You are working on detecting circular objects (e.g., coins) in an image, but the image has significant noise that interferes with object detection.

**Solution:**
*   Denoise the image using a Gaussian or median filter.
*   Apply edge detection (Canny Edge) to detect the outlines of the objects.
*   Use the Hough Circle Transform to detect the circular shapes.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('/content/coins.jpeg', cv2.IMREAD_GRAYSCALE)

blurred_img = cv2.GaussianBlur(img, (9, 9), 2)

edges = cv2.Canny(blurred_img, 50, 150)

circles = cv2.HoughCircles(blurred_img, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                           param1=100, param2=30, minRadius=15, maxRadius=50)

if circles is not None:
    circles = np.round(circles[0, :]).astype("int")

    output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for (x, y, r) in circles:
        cv2.circle(output, (x, y), r, (0, 255, 0), 4)

plt.figure(figsize=(5, 5))
plt.subplot(1, 2, 1)
plt.imshow(edges, cmap='gray')
plt.title('Edge Detected Image')

plt.subplot(1, 2, 2)
plt.imshow(output)
plt.title('Circles Detected')

plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/fa19da0b-6939-45bf-a24a-4ca64695e7aa)

**Scenario: Background Removal**

**Problem:**
You have an image of a product and you want to remove the background for a catalog or e-commerce listing, leaving only the product.

**Solution:**
*   Convert the image to grayscale.
*   Apply thresholding to create a binary mask.
*   Use the mask to remove the background from the original image.
  
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('/content/product.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

mask = np.zeros_like(gray)
cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
product_only = cv2.bitwise_and(img, img, mask=mask)

plt.figure(figsize=(5, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(product_only, cv2.COLOR_BGR2RGB))
plt.title('Background Removed')

plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/89b5f663-7ea1-4a0f-8db5-72318335824d)

**Scenario : Edge Detection for Shape Analysis**

**Problem:**
You want to detect the edges of objects in an image for shape analysis or to count distinct objects.

**Solution:**
*   Convert the image to grayscale.
*   Apply Canny Edge Detection to highlight edges.
*   Use contours to detect the shapes of the objects.

```python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('/content/camera.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_img = img.copy()
cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 3)

plt.figure(figsize=(5, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
plt.title('Contours Detected')

plt.tight_layout()
plt.show()
```

![image](https://github.com/user-attachments/assets/4a0c3817-140f-4e26-ae98-d21da049b918)

**Assignment:**

**Implementing Image Transformations and Filtering:**

*   Choose a set of images and apply the techniques you've learned, including scaling, rotation, blurring, and edge detection.
*   Documentation: Document the steps taken, and the results achieved in a report.

```python
image = cv2.imread('/content/trinidad.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

scale_percent = 50
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
resized_image = cv2.resize(image, dim)

angle = 45
center = (image.shape[1] // 2, image.shape[0] // 2)
rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

Gaussian = cv2.GaussianBlur(image, (5, 5), 0)
median = cv2.medianBlur(image, 5)
bilateral = cv2.bilateralFilter(image, 9, 75, 75)

fig, axes = plt.subplots(2, 3, figsize=(10, 5))

axes[0][0].imshow(image)
axes[0][0].set_title("Original Image")

axes[0][1].imshow(resized_image)
axes[0][1].set_title("Scaled Image")

axes[0][2].imshow(rotated_image)
axes[0][2].set_title("Rotated Image")

axes[1][0].imshow(Gaussian)
axes[1][0].set_title("Gaussian Filter")

axes[1][1].imshow(median)
axes[1][1].set_title("Median Filter")

axes[1][2].imshow(bilateral)
axes[1][2].set_title("Bilateral Filter")

plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/0c81396e-dcad-416c-90e8-0fee42860014)

# **Image Processing Report**

**Image loading and conversion:**

*   **Action:** Loaded the image and converted it from BGR to RGB.
*   **Purpose:** Ensured correct color representation for display.

**Resizing:**

*  **Action:** Scaled down the image to half its original size.
*  **Result:** Produced a smaller version of the image.

**Rotation:**

*   **Action:** Rotated the image by 45 degrees.
*   **Results:** Demonstrated how changes in orientation affect image appearance.

**Filtering:**

**Gaussian Blur:**

*   **Action:** Applied Gaussian blur to the image.
*   **Result:** slightly blurred the image, reducing granularity.

**Median Blur:**
*   **Action:** Applied median blur to the image.
*   **Results:** reduced salt-and-pepper noise while preserving edges.

**Bilateral Filter:**
*   **Action:** Used a bilateral filter on the image.
*   **Result:** Reduced noise and smoothed the image while maintaining edge details.

**Image States:**

*   **Original Image:** The base image before any processing.

*   **Scaled Image:** The resized (half-size) version of the image.

*   **Rotated Image:** The image rotated by 45 degrees.

*   **Gaussian Filtered Image:** The image after applying Gaussian blur.

*   **Median Filtered Image:** The image after applying median blur.

*   **Bilateral Filtered Image:** The image after applying a bilateral filter.


# **Submission Instruction:**

*  **Create Folder under the Github Repository of the subject.**

*  **Submission Format:**

*  Upload your processed images, code, and documentation to the GitHub repository.
*  Ensure all content is well-organized and clearly labeled.

**• Filename Format:** [SECTION-BERNARDINO-MP2] 4D-BERNARDINO-MP2

**Penalties:** Failure to follow these instructions will result in a 5-point deduction for incorrect filename format and a 5-point deduction per day for late submission. Cheating and plagiarism will be penalized.



