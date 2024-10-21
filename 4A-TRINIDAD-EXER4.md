Trinidad, Cyla Dendeyl M.
BSCS 4A
**CSST 106 (PERCEPTION AND COMPUTER VISION)**

# **Exercise 1: HOG (Histogram of Oriented Gradients) Object Detection**

**Task:**
HOG is a feature descriptor widely used for object detection, particularly for human detection. 

In this exercise, you will:
* Load an image containing a person or an object.
* Convert the image to grayscale.
* Apply the HOG descriptor to extract features.
* Visualize the gradient orientations on the image.
* Implement a simple object detector using HOG features.

**Key Points:**
* HOG focuses on the structure of objects through gradients.
* Useful for detecting humans and general object recognition.

 ```python
import cv2
from skimage.feature import hog
import matplotlib.pyplot as plt

image = cv2.imread('cylaaa.jpeg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

hog_features, hog_image = hog(
    gray_image, 
    orientations=9, 
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2), 
    block_norm='L2-Hys',
    visualize=True,
    feature_vector=True,
    )

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Grayscale Image')

plt.subplot(1, 2, 2)
plt.imshow(hog_image, cmap='gray')
plt.title('HOG Image')

plt.show()
 ```
![image](https://github.com/user-attachments/assets/8f847b65-9c96-4daa-b47a-8f3a673698e7)

**Explanation:**
The HOG image represents the distribution of edge orientations within the original image, which can be useful for object detection and recognition tasks.

# **Exercise 2: YOLO (You Only Look Once) Object Detection**

**Task:**
YOLO is a deep learning-based object detection method. 

In this exercise, you will:
* Load a pre-trained YOLO model using TensorFlow.
* Feed an image to the YOLO model for object detection.
* Visualize the bounding boxes and class labels on the detected objects in the image.
* Test the model on multiple images to observe its performance.

 ```python
!wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
!tar -zxf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
 ```

 ```python
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
layer_names = net.getLayerNames()

output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

image = cv2.imread('cat.jpg')
height, width, channels = image.shape

blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.7:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            label = f"{classes[class_id]}: {confidence:.2f}"
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2_imshow(image)
 ```
![image](https://github.com/user-attachments/assets/eb02d553-56fc-4a81-a228-badcfa403412)

**Explanation:**
In this exercise it performs object detection in an image using the YOLO algorithm. It loads the YOLO model, preprocesses the input image, runs the model, and draws bounding boxes around detected objects.

# **Exercise 3: SSD (Single Shot MultiBox Detector) with TensorFlow**

**Task:** SSD is a real-time object detection method. For this exercise:
* Load an image of your choice.
* Utilize the TensorFlow Object Detection API to apply the SSD model.
* Detect objects within the image and draw bounding boxes around them.
* Compare the results with those obtained from the YOLO model.

**Key Points:** 
* SSD is efficient in terms of speed and accuracy.
* Ideal for applications requiring both speed and moderate precision.

 ```python
import tensorflow as tf
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

model = tf.saved_model.load('ssd_mobilenet_v2_coco_2018_03_29/saved_model')

def run_inference_for_single_image(model, image):
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]

    output_dict = model.signatures['serving_default'](input_tensor)

    return output_dict

image_path = 'cat.jpg'
image_np = cv2.imread(image_path)

detections = run_inference_for_single_image(model, image_np)

num_detections = int(detections.pop('num_detections'))
detections = {key:value[0, :num_detections].numpy()
              for key,value in detections.items()}
detections['num_detections'] = num_detections

image_np_with_detections = image_np.copy()
for i in range(num_detections):
    if detections['detection_scores'][i] > 0.5:
        ymin, xmin, ymax, xmax = detections['detection_boxes'][i]
        (left, right, top, bottom) = (xmin * image_np.shape[1], xmax * image_np.shape[1], ymin * image_np.shape[0], ymax * image_np.shape[0])
        cv2.rectangle(image_np_with_detections, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)

cv2_imshow(image_np_with_detections)
```
![image](https://github.com/user-attachments/assets/036c4bc6-502c-4d3f-9c8d-e9e9cd28f442)

**Explanation:**
Single-shot multibox detection (SSD) generates a varying number of anchor boxes with different sizes, and detects varying-size objects by predicting classes and offsets of these anchor boxes (thus the bounding boxes); thus, this is a multiscale object detection model.

# **Exercise 4: Traditional vs. Deep Learning Object Detection Comparison**
**Task:**
Compare traditional object detection (e.g., HOG-SVM) with deep learning-based methods (YOLO, SSD):
* Implement HOG-SVM and either YOLO or SSD for the same dataset.
* Compare their performances in terms of accuracy and speed.
* Document the advantages and disadvantages of each method.

**Key Points:**
* Traditional methods may perform better in resource-constrained environments.
* Deep learning methods are generally more accurate but require more computational
power.

```python
import cv2
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#HOG-SVM Implementation

# Initialize HOG descriptor and set SVM detector (pre-trained for pedestrian detection)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def detect_hog(image_path):
    """Detect objects using HOG-SVM."""
    # Load and resize image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (640, 480))
    
    # Detect objects (e.g., pedestrians)
    start_time = time.time()
    boxes, _ = hog.detectMultiScale(image, winStride=(8, 8), padding=(8, 8), scale=1.05)
    detection_time = time.time() - start_time

    # Draw bounding boxes
    for (x, y, w, h) in boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image, detection_time

#SSD (Deep Learning) Implementation

model = tf.saved_model.load('ssd_mobilenet_v2_coco_2018_03_29/saved_model')

def run_inference_for_single_image(model, image):
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]

    output_dict = model.signatures['serving_default'](input_tensor)

    return output_dict

def detect_ssd(image_path):
    """Detect objects using SSD MobileNet V2."""
    # Load and preprocess the image
    image_np = cv2.imread(image_path)
    image_resized = cv2.resize(image_np, (300, 300))  # SSD expects 300x300 input size
    
    # Run SSD inference
    start_time = time.time()
    detections = run_inference_for_single_image(model, image_resized)
    detection_time = time.time() - start_time
    
    # Process detections
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections
    
    # Draw bounding boxes on the image
    image_np_with_detections = image_np.copy()
    for i in range(num_detections):
        if detections['detection_scores'][i] > 0.5:  # Draw boxes with confidence > 0.5
            ymin, xmin, ymax, xmax = detections['detection_boxes'][i]
            (left, right, top, bottom) = (xmin * image_np.shape[1], xmax * image_np.shape[1], ymin * image_np.shape[0], ymax * image_np.shape[0])
            cv2.rectangle(image_np_with_detections, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)

    return image_np_with_detections, detection_time

#Performance Comparison

# Define a list of test images
image_paths = ['cylaaa.jpeg', 'cat.jpg']  # Add your image paths

# Initialize lists to store timings
hog_times = []
ssd_times = []

# Perform detection for each image and compare performance
for image_path in image_paths:
    # HOG-SVM detection
    hog_detected_image, hog_time = detect_hog(image_path)
    hog_times.append(hog_time)

    # SSD detection
    ssd_detected_image, ssd_time = detect_ssd(image_path)
    ssd_times.append(ssd_time)

    # Plot side by side for comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Show HOG-SVM result
    axes[0].imshow(cv2.cvtColor(hog_detected_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"HOG-SVM Detection - Time: {hog_time:.4f}s")
    axes[0].axis('off')

    # Show SSD result
    axes[1].imshow(cv2.cvtColor(ssd_detected_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"SSD Detection - Time: {ssd_time:.4f}s")
    axes[1].axis('off')

    plt.show()

# Compute average timings
avg_hog_time = np.mean(hog_times)
avg_ssd_time = np.mean(ssd_times)

# Display the comparison results in a neatly formatted manner
print(f"{'Method':<15} {'Avg Detection Time (s)':<25}")
print(f"{'-'*40}")
print(f"{'HOG-SVM':<15} {avg_hog_time:.4f}")
print(f"{'SSD':<15} {avg_ssd_time:.4f}")
```
![image](https://github.com/user-attachments/assets/e1e73411-0bcb-4fb2-bded-c117e1cbf50e)
![image](https://github.com/user-attachments/assets/7777e7d2-ce4a-4f26-b6f4-db5415ee03c0)

Method          Avg Detection Time (s)   
HOG-SVM         0.2206
SSD             2.7473

**HOG-SVM** is a good choice for simpler, resource-constrained environments where speed is important and computational power is limited. However, it lacks the flexibility and accuracy needed for more complex object detection tasks.

**Deep learning (SSD/YOLO)** offers superior accuracy and versatility, making it ideal for complex environments and tasks involving diverse object categories. The trade-off is higher computational requirements, making them better suited for systems with GPU support or more powerful hardware.


