
# CSST106-CS4A
Perception and Computer Vision - Machine Problem 1 

https://github.com/user-attachments/assets/a327272a-fbda-4bed-9cdb-8076af1e6dd8

# **Introduction to Computer Vision**

  Computer vision is a field of artificial intelligence (AI) that uses machine learning and neural networks to teach computers and systems to derive meaningful information from digital images, videos, and other visual inputs—and to make recommendations or take actions when they see defects or issues. It has key concepts, including:

* Image Acquisition
* Image Preprocessing
* Feature Extraction
* Feature Description
* Object Detection
* Object Tracking
* Image Segmentation
* Image Classification

From the perspective of AI, the main way of understanding images comes from deep learning technology built on so-called convolutional neural networks (CNNs). Neural networks are modeled after the biological processes that occur in human brains, particularly the way the visual cortex functions.

In 2023, the enhanced performance of CNN architectures has increased the effectiveness and accuracy of visual recognition problems, resulting in AI being able to represent shapes and images exactly. As of 2024, AI models can not only detect individual items in a scene and classify them correctly, but they can also interpret more complicated images considering the context and relationships involved in a scene. One explanation of the astonishing speed with which AI has developed in terms of visual understanding is the fact that large-scale, annotated datasets have come into existence.

# **Roles that AI plays in image processing**

**Image classification:** using machine learning and deep learning algorithms and the like to group images or tag them. This permits applications like automated sorting, recognition, and scene understanding.

**Object detection:** visualizing and determining objects in images. Labeled object datasets are used to teach models the object boundaries; this is how it becomes. It's equivalent to line drawings of semantic objects shown here of a base object, whether the road or a person. A combination of the comvolutional (one of the strongest deep learning algorithms) neural networks, one of the most effective artificial intelligence-based tools for pixel-by-pixel labeling of images, plays the leading role.

**Image enhancement:** application of AI models optimized to the images, such as super-resolution, noise removal, contrast adjustment, and color correction. The latter quality improvement is a prerequisite for efficient image usage.

**Image generation:** Sometimes the creation of artificial data is not only cool but also practical, or it might, as a template, be used to enrich training sets or build problems in an advanced way.

**Anomaly detection:** Figuring out the images that are abnormal with the help of unsupervised learning, which doesn't have labeled data. It is the easiest way to generate a new prototype.

**Visual recommendation:** ensuring that users can watch specific pictures or movies in the form they are most likely to like. Likewise, these effects are accomplished with deep systems and reinforcement learning.

**Content moderation:** spotting inappropriate or threatening visual matter such as nudity, demotivation, and negative words using the methods of dataset training to learn from a vast array of visual contexts. 

  Image processing is the core of AI, and it gives the AI the ability to learn and interact with the visual environment. This can be achieved through the application of image transformation, image manipulation, and image analysis. AI can thus implement such tasks as object recognition, scene understanding, data analysis, content creation, and human-computer interaction.

  A case in point is medical imaging, where AI can be employed to probe X-rays, MRIs, and CT scans for abnormal areas, thus helping the physician in his diagnosis. Autonomous cars utilize AI image processing to detect the location of cars, pedestrians, and road signs; hence, the AI can safely maneuver through these elements. Content creation is one domain where AI has the potential to produce very realistic images and the deepfake generation of film content. However, the question of the importance of ethical considerations in image processing can be raised as well.

# **Overview of Image Processing Techniques**

**Key Techniques Used in Image Processing**

**Filtering:** Linear image filtering techniques that include Sobel, Gaussian, and mean filters are used to achieve different goals in image processing. Sobel filters are the most important for edge detection, while Gaussian filters are suitable for noise reduction. The mean filter, however, replaces each pixel with the average of its neighbors. These filters can encounter problems in instances of non-linear noise patterns or when the intricate details are involved.

**Edge detection:** Edge detection is an important procedure with high priority in the area of computer vision and image processing technology. It is the technique of locating sharp discontinuities in an image, which usually correspond to intensity or color change. These discontinuities are called edges, and they are of great importance in the analysis of the structure and contents of an image.

**Segmentation:** Image Segmentation in the Image Process. It discussed one of the key computer vision tasks and how this process helps image processing and analysis in many different fields, including medical image analytics for diagnosis and planning better treatment methods. Also, the traditional image segmentation models overshadow how advanced deep learning models are used today in image processing and segmentation tasks.

**Three core techniques**

**Convolutional neural networks (CNNs)** are a class of neural networks that have been purposely designed to process data related to images. They first use filters to detect certain features in diagrams like corners, textures, or edges and then join these features to create a higher-level representation of the diagram.

CNNs come with a lot of pros, including:
* Feature Learning
* Hierarchical representation
* High accuracy
  
**Transfer learning** is when a model is initially trained on a large dataset containing a lot of images and then it is fine-tuned on a smaller, more specific dataset. Thus, the model can use the knowledge it acquired from the large dataset to do better on the smaller dataset.

The advantages of transfer learning are:
* Reduced training time
* Better performance
* Less data requirements
  
**Attention mechanisms** are what allow AI systems to focus only on certain parts of an image that are most important to the task at hand; thus, the system collects more relevant data from the image. Moreover, it enables the system to extract truly informative data from the image.

Benefits of attention mechanisms are such as:
* Improved performance
* Interpretability
* Efficiency

# **Facial Recognition system**

  A facial recognition system is an excellent part of the computer vision industry, widely used in mobile phones and security systems. The technique has become most popular, replacing password logins for users in their daily lives.

**Image Processing Techniques in Facial Recognition Systems**

**Color images** often contain background clutter that reduces the accuracy of face detection and facial recognition systems. To improve this, they have methods to remove that unneeded color data from a color input image before recognition performs on the clear grayscale version. The result was efficient, fast, and accurate processing of millions of facial images for the application.

**Image cropping** removes unnecessary surrounding material from the images for some specific reason. Image post-processing can help to extract relevant data. For example, many extraction methods are used in face detection systems to ensure the face in the image crop is in the most suitable position.

**Image filtering algorithms** reduce the effect of noise on the image. As a result, image filtering improves gray-level coherence, background white noise, and smoothness. In addition, the regularized inverse auto-regressive (RIR) filter also results in a sharpened output image.

  In facial recognition systems, accurately detecting the presence of a face within an image is a critical first step. This involves locating the face and distinguishing it from other objects in the background. A common challenge is ensuring that the algorithm can identify faces under varying conditions, such as different lighting, angles, or occlusions.

  An edge detection algorithm refers to a technique used in image analysis and computer vision to identify the locations of significant edges in an image while filtering out false edges caused by noise. It involves applying a high-pass filter to measure the rate of change at each pixel and then thresholding the filter output to determine the pixels that represent edges. The algorithm can be further enhanced by applying techniques like edge thinning to precisely locate the edges in the image.

``` python

import cv2
import matplotlib.pyplot as plt

image_path = '/content/finalface1.png'
image = cv2.imread(image_path)

if image is None:
    raise ValueError(f"Image at path '{image_path}' could not be loaded.")

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

edges = cv2.Canny(blurred_image, 100, 200)
edges_colored = cv2.applyColorMap(edges, cv2.COLORMAP_HOT)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(1, 3, 2)
plt.title('Grayscale Image')
plt.imshow(gray_image, cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Edge Detection')
plt.imshow(edges_colored)

plt.tight_layout() 
plt.show()

![image](https://github.com/user-attachments/assets/fe45df7b-88f6-4ee6-a111-e21c26ae1517)


```
# **Research an Emerging Form of Image Processing**

**Fourier Neural Operators (FNO)** are an innovative deep learning technique aimed at solving high-dimensional, complex problems such as those found in scientific computing, including fluid dynamics, weather prediction, and more. This method uses neural networks in conjunction with Fourier transforms to model partial differential equations (PDEs) efficiently. FNOs are emerging as a powerful tool for both scientific applications and broader AI-based image processing tasks.

**Potential Impact on Future AI Systems**

* Scientific Simulations
* Medical Imaging and Diagnostics:
* Autonomous Systems
* Environmental Modeling
* Engineering and Manufacturing
* Image Processing













