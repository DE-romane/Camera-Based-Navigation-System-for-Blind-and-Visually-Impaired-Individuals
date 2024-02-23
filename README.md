# Camera Based Navigation System for Blind and Visually Impaired Individuals

## Team Members
**Romany Tawfeek Aziz**
**Islam Mohamed Kamal**
**Karam Nasreddin Abulhassan**
**Mahmoud Ahmed Ibrahim**
**Mohamed Fareed Gad**
**Mohamed Saber El Sayed**

**Supervised by :**
**Dr. Mostafa Salah**
**Dr. Ahmed Farghal**

## Introduction
In this project, we present a camera-based navigation system for blind and visually impaired people. The system consists of a pair of smart glasses equipped with a camera, a Raspberry Pi, an ultrasonic sensor, and a headphone. The system can perform various tasks such as optical character recognition, text to speech, face recognition, object detection, and obstacle avoidance. The system aims to help the users to interact with their surroundings more easily and safely. We describe the hardware and software components of the system, the design and implementation of the deep learning models, and the results of the evaluation.

![[Introduction-1.jpg|500]]

![[Introduction-2.jpg|500]]

## System prototype 
![[System-prototype.png]]
## Features
![[Features-1.png]]

Users can switch between different modes using physical buttons connected to the Raspberry Pi's GPIO pins.

- **Object Detection Mode (0):** Detects and identifies objects in the camera's field of view using deep learning models. The system can recognize various objects and provide audio feedback about their presence and location relative to the user.
- **Face Recognition Mode (1):** Identifies faces of known individuals from a pre-trained dataset and announces their names. This mode assists in recognizing familiar faces and enhancing social interactions.
- **OCR (Optical Character Recognition) Mode (2):** Extracts text from images captured by the camera and converts it into audible speech. This feature enables users to read text from signs, documents, or other printed materials.

![[Features-2.jpg]]

### Object Detection Mode (Mode 0)

The Object Detection Mode involves distance calculation and ensemble learning-based models for obstacle avoidance and object detection.

**Distance Calculation:**
- Distance estimation is performed using an ultrasonic sensor.
- Calculations occur at multiple frames to optimize power consumption.
- Distance is calculated using the round-trip time for the triggered wave.
- A threshold of 30 centimeters triggers a stop alert for obstacle avoidance.

**Deep Learning for Object Detection:**
- Utilization of pre-trained SSD model A for general object detection.
- Fine-tuning of model A to adapt to specific classes using transfer learning.
- Integration of a custom object detection model, Model B, to detect objects not covered by Model A.

**Data Collection and Preparation for Model B:**
- Collection of images from various sources including CIFAR-100, CALTECH-101, and manually captured images.
- Images resized to 1024 by 1024 JPEG for preprocessing.
- Data imbalance addressed by gathering 60 images per class for Model B.
- Annotation of images using LabelImg tool to generate XML files containing object information.

**Training and Deployment of Model B:**
- Implementation of custom object detection models using TensorFlow 2 detection model zoo.
- Utilization of SSD architecture for object detection.
- Training conducted on Google Colaboratory with GPU access.
- Metrics monitored using TensorBoard with focus on total loss, mean average precision, and average recall.

### Face Recognition Mode (Mode 1)

The Face Recognition Mode involves detecting and recognizing human faces from input video frames.

**Face Detection and Recognition:**
- Face detection separates faces from the background using pre-processing techniques.
- Face recognition module verifies detected faces against known faces in a database.
- Dataset collected for training containing images of individuals named (Romany ,Karam, Badran, and Fareed).
- Training performed using Python scripts.

### Reading Text Mode (Mode 2)

The Reading Text Mode utilizes Optical Character Recognition (OCR) to detect and read text from images or video frames.

**OCR Implementation:**
- Utilization of the Tesseract-OCR Engine through Python-tesseract tool.
- Pre-processing techniques such as grayscale conversion and noise removal can enhance OCR accuracy.
- Testing conducted on both images and video frames, with satisfactory results observed for larger texts.

### Optimization and Resource Utilization

Various optimizations were implemented to enhance performance and resource utilization across all modes:

**Optimizations Implemented:**
- Multi-threading used to improve frame capture speed.
- Reduction of redundant information conveyed to the user, such as object counts instead of repeating object names.
- Ultrasonic sensor code optimized for higher frequency operation in the absence of detected objects.
- Minor optimizations applied to replace repeated logical statements with single arithmetic operations, reducing computational overhead.

## Hardware 
The hardware used in this project are:

 - **Raspberry Pi 4 Model B+**: A single-board computer that runs the main software of the system, such as the deep learning models, the computer vision algorithms, and the audio feedback. It has a 40-pin GPIO header, 4 USB ports, an Ethernet port, and a 3.5mm audio jack
 
- **Raspberry Pi Camera**: A camera module that can capture high definition video and still images. It is connected to the Raspberry Pi via a ribbon cable and is used to recognize objects, faces, and text in the environment

- **Battery**: A power supply for the Raspberry Pi and the camera module. It should have enough capacity to run the system for a reasonable amount of time

- **SD card**: A storage device for the Raspberry Pi that contains the operating system (Raspbian OS) and the files needed for the system. It should have at least 8GB of space

- **Ultrasonic Sensor**: A sensor that uses sound waves to measure the distance between the sensor and the nearest object. It is connected to the Raspberry Pi via the GPIO pins and is used to detect obstacles and warn the user

- **Headphone**: A device that delivers audio feedback to the user via speech synthesis. It is connected to the Raspberry Pi via the audio jack and is used to inform the user about the recognized objects, faces, or text

## Software Dependencies
- Python 3.x
- OpenCV
- TensorFlow Lite
- Face Recognition Library
- pyttsx3 (Text-to-Speech)
- RPi.GPIO (for GPIO operations)
## 3D printing 
**3D Printing (Additive Manufacturing):****
- Definition: 3D printing or additive manufacturing is the process of creating three-dimensional solid objects from a digital file.
- Process: Objects are built layer by layer using additive processes, where material is deposited in successive layers to form the final object.
- Contrast with Subtractive Manufacturing: Unlike subtractive manufacturing, which involves cutting or hollowing out material, 3D printing adds material to create objects.
- Advantage: 3D printing enables the production of complex shapes using less material compared to traditional manufacturing methods.

**CAD Software for 3D Printing:**
- Recommendation: Autodesk Fusion360 is preferred for designing and creating efficient mechanical parts in 3D printing.
- Description: Fusion360 offers a comprehensive suite of tools for parametric modeling, assembly design, simulation, and computer-aided manufacturing.
- File Format: STL (Standard Triangle Language) is the native file format for stereolithography CAD software by 3D Systems, widely used in rapid prototyping, 3D printing, and computer-aided manufacturing.
![[3Dprinting-1.jpg]]

![[3Dprinting-2.jpg]]

![[3Dprinting-3.jpg]]

![[3Dprinting-4.jpg]]

![[3Dprinting-5.jpg]]

![[3Dprinting-6.jpg]]

![[3Dprinting-7.jpg]]

![[3Dprinting-8.jpg]]

## User Interaction
- Users can switch between different modes using physical buttons connected to the Raspberry Pi's GPIO pins.
- Audio feedback is provided in real-time, allowing users to navigate their surroundings and interact with the system seamlessly.

## Future Enhancements
- Integration with additional sensors for improved spatial awareness and obstacle detection.
- Incorporation of machine learning algorithms to enhance object detection accuracy and expand the range of recognized objects.
- Development of a mobile application interface for remote control and configuration of the navigation system.

## Conclusion
The Camera Based Navigation System offers a valuable tool for blind and visually impaired individuals to navigate their environment independently. By leveraging advanced technologies such as deep learning and real-time image processing, the system provides crucial information about objects, faces, and text, enabling users to interact with the world more confidently and autonomously.


![[IMG_20220714_235207_919.jpg]]




![[IMG_20220714_235204_349.jpg]]





## resource

code :

the presentation :
