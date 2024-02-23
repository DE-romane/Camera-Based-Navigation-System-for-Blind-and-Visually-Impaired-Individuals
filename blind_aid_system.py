# import the necessary Libraries
import os
import argparse # For parsing command-line arguments.
import numpy as np # For numerical computing with arrays and matrices.
import sys 
from threading import Thread # Allows multiple functions to execute concurrently.
import importlib.util #Provides utilities for working with import-related functionality.
from imutils.video import VideoStream # for handling video streams from the webcam.
import face_recognition
import imutils
import pickle
import time
import cv2
import pyttsx3 #* text to speech library
import RPi.GPIO as GPIO
import enum


### Initialize pyttsx3 engine
engine = pyttsx3.init()
#Retrieves the current speaking rate.
rate = engine.getProperty('rate') 
#Sets the speaking rate to a reduced value.
engine.setProperty('rate', rate-50)
#Retrieves available voices.
voices = engine.getProperty('voices')
#Sets the voice to the fourth available voice.
engine.setProperty('voice', voices[3].id)


#to run the pyttsx3 engine in another thread using engine.runAndWait().
def run_pyttsx3_engine():
    engine.runAndWait()



#mainly used for ultrasonic trig and echo pins and the pushbuttons
GPIO.setmode(GPIO.BOARD)

# ultrasonic pins
TRIG = 16 #trigger pin
ECHO = 18 #echo pin
GPIO.setup(TRIG,GPIO.OUT)
GPIO.setup(ECHO,GPIO.IN)

# pushbuttons pins
button0 = 12
button1 = 13
button2 = 14
GPIO.setup(button0, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(button1, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(button2, GPIO.IN, pull_up_down=GPIO.PUD_UP)


# Using enum class create enumerations for the modes
class Modes(enum.Enum):
   object_detection = 0
   face_recognition = 1
   OCR = 2
#sets the initial mode of operation to be 
curr_mode = Modes.object_detection 


######## Object detection mode initializations ############
# Define VideoStream class to handle streaming of video from webcam in separate processing thread

class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=10):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        #set the video capture properties for codec 
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
        
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

    # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
    # Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
    # Return the most recent frame
        return self.frame

    def stop(self):
    # Indicate that the camera and thread should be stopped
        self.stopped = True

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graphA', help='Name of the .tflite file of modelA, if different than detect.tflite',
                    default='modelA_detect.tflite')
parser.add_argument('--graphB', help='Name of the .tflite file of modelB',
                    default='modelB_detect.tflite')
parser.add_argument('--labelsA', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap_A.txt')
parser.add_argument('--labelsB', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap_B.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.45)    #    You may want to change that
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='640x480')    #    You may want to change that also
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')
parser.add_argument('--detectlimit', help='minimum number of object detections per sound frame',
                    default=3)     #* Put certain limit to detections utilizing sound and NOTE: you have to change '#%' statements
parser.add_argument('--decisionframes', help='the number of frames that decision is taken and sounds should be outputed',
                    default=37)     #* Put the rate at which the sound be outputed in frames
parser.add_argument('--ultrasonicframes', help='the number of frames that ultraonic will give reading',
                    default=7)     #* Put the rate of ulltrasonic readings in frames

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME_A = args.graphA
GRAPH_NAME_B = args.graphB
LABELMAP_NAME_A = args.labelsA
LABELMAP_NAME_B = args.labelsB
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu
DETECT_LIMIT = args.detectlimit
DECISION_FREQUENCY =args.decisionframes
#ULTRASONIC_FREQUENCY = args.ultrasonicframes

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

#If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'): #* change this name if needed
        GRAPH_NAME = 'edgetpu.tflite'       

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT_A = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME_A)
PATH_TO_CKPT_B = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME_B)
# Path to label map file
PATH_TO_LABELS_A = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME_A)
PATH_TO_LABELS_B = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME_B)
# Load the label map for model A
with open(PATH_TO_LABELS_A, 'r') as f:
    labelsA = [line.strip() for line in f.readlines()]
# Load the label map for model B
with open(PATH_TO_LABELS_B, 'r') as f:
    labelsB = [line.strip() for line in f.readlines()]
# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labelsA[0] == '???':
    del(labelsA[0])
if labelsB[0] == '???':
    del(labelsB[0])
# Load the Tensorflow Lite model A .
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter_A = Interpreter(model_path=PATH_TO_CKPT_A,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT_A)
else:
    interpreter_A = Interpreter(model_path=PATH_TO_CKPT_A)


interpreter_A.allocate_tensors()

# Get model A details
input_details_A = interpreter_A.get_input_details()
output_details_A = interpreter_A.get_output_details()
height_A = input_details_A[0]['shape'][1]
width_A = input_details_A[0]['shape'][2]

floating_model_A = (input_details_A[0]['dtype'] == np.float32)

interpreter_A.allocate_tensors()
input_mean = 127.5
input_std = 127.5

#* Load TFlite of model B
if use_TPU:
    interpreter_B = Interpreter(model_path=PATH_TO_CKPT_B,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT_B)
else:
    interpreter_B = Interpreter(model_path=PATH_TO_CKPT_B)
    
interpreter_B.allocate_tensors()

# Get model B details
input_details_B = interpreter_B.get_input_details()
output_details_B = interpreter_B.get_output_details()
height_B = input_details_B[0]['shape'][1]
width_B = input_details_B[0]['shape'][2]

floating_model_B = (input_details_B[0]['dtype'] == np.float32)
interpreter_B.allocate_tensors()


# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=10).start()
time.sleep(1.5)

# loop over frames from the video file stream
translateCount = {1: 'one',2: 'two',3:'three',4: 'four',5: 'five',6: 'six',7: 'seven',8: 'eight',9: 'nine'} # maybe you only need 3

detection_limit = DETECT_LIMIT
# Initialize the counter for each object for mode A
detection_counter = {labelsA[i] : 0 for i in range(len(labelsA))}

third_imW = imW/3 #* Dividing the screen into three parts
 #* Every DECISION_FREQUENCY (33) frames we will output the audio
decision_frames = 0 #* Count the frames till it reaches the DECISION_FREQUENCY then resets
#ultraSonic_frames_freq = ULTRASONIC_FREQUENCY #* Every ULTRASONIC_FREQUENCY (7) frames we will get an ultrasonic reading. NOTE : if you change this you should change it
#ultraSonic_frames = 0 #* Count the frames till it reaches the ultraSonic_frames_freq then resets

######## Face recognition mode initializations ###########
# variables Initialize 'currentname' to trigger only when a new person is identified.
currentname = "unknown"
#Determine faces from encodings.pickle file model created from train_model.py
encodingsP = "encodings.pickle"
#use this xml file
#https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
cascade = "haarcascade_frontalface_default.xml"

# load the known faces and embeddings along with OpenCV's Haar
# cascade for face detection
#print("[INFO] loading encodings + face detector…")
data = pickle.loads(open(encodingsP, "rb").read())
detector = cv2.CascadeClassifier(cascade)

########## OCR (reading text) mode dependencies  ####################
import pytesseract
from pytesseract import Output

#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
while True:
    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()
    # Mode selection
    # if GPIO.input(button0) == false:
        #curr_mode = Modes.object_detection
        #engine.say('Object detection mode')
        #engine.runAndWait()
    # elif GPIO.input(button1) == false:
        #curr_mode = Modes.face_recognition
        #engine.say('face recognition mode')
        #engine.runAndWait()
    # elif GPIO.input(button2) == false:
        #curr_mode = Modes.OCR
        #engine.say('Reading text mode')
        #engine.runAndWait()
###################################    Mode 0    ##############################
    if curr_mode == Modes.object_detection:
        # Grab frame from video stream
        frame1 = videostream.read()
        #* claculate the actual number of detected objects that heared by the user
        numberOfobjects_Heared = 0
        # Acquire frame and resize to expected shape [1xHxWx3]
        frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width_A, height_A))
        input_data_A = np.expand_dims(frame_resized, axis=0)
        frame_resized = cv2.resize(frame_rgb, (width_B, height_B))
        input_data_B = np.expand_dims(frame_resized, axis=0)
        
        
        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model_A:
            input_data_A = (np.float32(input_data_A) - input_mean) / input_std
        if floating_model_B:
            input_data_B = (np.float32(input_data_B) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input for model A
        interpreter_A.set_tensor(input_details_A[0]['index'],input_data_A)
        interpreter_A.invoke()
        # Perform the actual detection by running the model with the image as input for model B
        interpreter_B.set_tensor(input_details_B[0]['index'],input_data_B)
        interpreter_B.invoke()
        
        # Retrieve detection results from model A
        A_boxes = interpreter_A.get_tensor(output_details_A[0]['index'])[0] # Bounding box coordinates of detected objects
        A_classes = interpreter_A.get_tensor(output_details_A[1]['index'])[0] # Class index of detected objects
        A_scores = interpreter_A.get_tensor(output_details_A[2]['index'])[0] # Confidence of detected objects # Notice the list is sorted
        #A_num = interpreter_A.get_tensor(output_details_A[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)
        
        # Retrieve detection results from model B
        B_boxes = interpreter_B.get_tensor(output_details_B[1]['index'])[0] # Bounding box coordinates of detected objects
        B_classes = interpreter_B.get_tensor(output_details_B[3]['index'])[0] # Class index of detected objects
        B_scores = interpreter_B.get_tensor(output_details_B[0]['index'])[0] # Confidence of detected objects # Notice the list is sorted
        #B_num = interpreter_B.get_tensor(output_details_B[2]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

        #* looping over full detections using len(scores) and draw detection box for each if confidence is above minimum threshold
        for i in range(len(A_scores)):
            if (A_scores[i] > min_conf_threshold):
                object_name = labelsA[int(A_classes[i])] #* Look up object name from "labelsA" array using class index
            
                #* if this is the frame of outputing audio
                if decision_frames % DECISION_FREQUENCY == 0:
                    detection_counter[object_name]+=1    #  increase the count of detected object
                # Get bounding box coordinates and draw box
                # For model A, Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin_A = int(max(1,(A_boxes[i][0] * imH)))
                xmin_A = int(max(1,(A_boxes[i][1] * imW)))
                ymax_A = int(min(imH,(A_boxes[i][2] * imH)))
                xmax_A = int(min(imW,(A_boxes[i][3] * imW)))
                cv2.rectangle(frame, (xmin_A,ymin_A), (xmax_A,ymax_A), (10, 255, 0), 2)

                # Draw label            
                label = '%s: %d%%' % (object_name, int(A_scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin_A, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin_A, label_ymin-labelSize[1]-10), (xmin_A+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin_A, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
        # Detection for model B
        for i in range(len(B_scores)):
            if (B_scores[i] > min_conf_threshold):
                object_name = labelsB[int(B_classes[i])] #* Look up object name from "labelsA" array using class index
                # Get bounding box coordinates and draw box
                # For model B, Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin_B = int(max(1,(B_boxes[i][0] * imH)))
                xmin_B = int(max(1,(B_boxes[i][1] * imW)))
                ymax_B = int(min(imH,(B_boxes[i][2] * imH)))
                xmax_B = int(min(imW,(B_boxes[i][3] * imW)))
                cv2.rectangle(frame, (xmin_B,ymin_B), (xmax_B,ymax_B), (255, 0, 0), 2)

                # Draw label            
                label = '%s: %d%%' % (object_name, int(B_scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin_B, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin_B, label_ymin-labelSize[1]-10), (xmin_B+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin_B, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

        #* Output sound based on the label
        #* if this is the decision frame we want to output the sound
        if(decision_frames % DECISION_FREQUENCY == 0):
            decision_frames = 0 #* reset the decision_frames
            B_uniqueClasses = list(set(B_classes)) #* get the unique detected B_classes
            A_uniqueClasses = list(set(A_classes)) #* get the unique detected A_classes
            A_uniqueScores = ['']*10
            i=0
            while (i < (len(A_uniqueClasses))):
                A_uniqueScores[i] =A_scores[list(A_classes).index(A_uniqueClasses[i])] #* get the unique scores corresponding to those unique classes
                if(A_uniqueScores[i] < min_conf_threshold):
                    del(A_uniqueScores[i])
                    del(A_uniqueClasses[i])
                    i-=1
                i+=1
            detection_limit = min(DETECT_LIMIT,len(A_uniqueClasses)) #* the detection limit is set to 'DETECT_LIMIT'(3) minimum if there is more than 'DETECT_LIMIT'(3) unique classes 
            First_detection_done = False #* this variable is used to detect first object so the string will contain 'and'
            for i in range(detection_limit):
                object_name = labelsA[int(A_uniqueClasses[i])]
                object_count = detection_counter[object_name]
                if object_count != 0:
                    if First_detection_done == True:
                        engine.say('annd')
                    engine.say(translateCount[object_count])
                    if object_count == 1:
                        engine.say(object_name)
                    else:
                        engine.say(object_name+'s')
                    numberOfobjects_Heared +=1
                    xmin = int(max(1,(A_boxes[i][1] * imW)))   #* determines left wall of BB
                    xmax = int(min(imW,(A_boxes[i][3] * imW))) #* determines right wall of BB
                    if(xmax > third_imW*2 and xmin > third_imW*2):
                        engine.say('on your right.')
                    elif(xmax < third_imW and xmin < third_imW):
                        engine.say('on your left.')
                    else:
                        engine.say('on your front.')
                First_detection_done=True
                detection_counter[object_name] = 0
            for i in range(detection_limit):
                object_name = labelsB[int(B_uniqueClasses[i])]
                if First_detection_done == True:
                    engine.say('and')
                engine.say(object_name)
                numberOfobjects_Heared+=1
                First_detection_done=True
            engine.say(' detected')
            Thread(target=run_pyttsx3_engine,args=(),daemon=True).start() # thread usage for generating the audio
            #* This if_else statement makes ultrasonic more effective if there is no object detected by the camera
     #       if(numberOfobjects_Heared == 0):
      #          ultraSonic_frames_freq = 2
        #    else:
         #       ultraSonic_frames_freq = ULTRASONIC_FREQUENCY
          #  decision_frames = DECISION_FREQUENCY-7
            #* These statements to make up for the short time pronouncing the sentence. NOTE : Depends on FPS
            #% Based on FPS and number of objects you want to detect change equation parameters the equation
            #% we want 6 objects if objects = 1 -> 19, if 2-> 16, if 3->12, if 4->9, if 5->5, if 6-> 1
            #% doing a pearson straight line fitting A*(obj)+B we can find A = -3.6, B = 22.933
            #% we will add 0.5 to ceil the value instead we make B = 23.433
            decision_frames = int(abs(-3.6*(numberOfobjects_Heared)+23.433))
        #* for calculating the frames for audio output
        decision_frames+=1
        
        #* ultrasonic part
        #* if this is the frame we want to ultrasonic to run
        #if(ultraSonic_frames % ultraSonic_frames_freq ==0):
         #   ultraSonic_frames = 0
          #  GPIO.output(TRIG, True)  # To make trigger transmit power
           # time.sleep(0.00001)  # For this time
            #GPIO.output(TRIG, False)  # Off trigger

            #while GPIO.input()==0: # When echo didn't receive take time
             #  startTime = time.time()

            #while GPIO.input(ECHO)==1: # When echo  receive take time
             #  endTime = time.time()

            #timeElapsed= endTime - startTime

            #distance = (timeElapsed * 34300) / 2    #Speed of voice by cm 34300/2
         
            #if distance < 30:
             #   os.system("mpg321 Sounds/stop.mp3")
             #   decision_frames = DECISION_FREQUENCY -7
        
        #* for calculating the frames for ultrasonic
        #ultraSonic_frames+=1
        
        # Draw framerate in corner of frame
        cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame)

###################################    Mode 1    ##############################
    elif curr_mode == Modes.face_recognition:
        # grab the frame from the threaded video stream and resize it
        # to 500px (to speedup processing)
        frame = videostream.read()
        frame = imutils.resize(frame, width=1000)
        
        # convert the input frame from (1) BGR to grayscale (for face
        # detection) and (2) from BGR to RGB (for face recognition)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # detect faces in the grayscale frame
        rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
            minNeighbors=5, minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE)

        # OpenCV returns bounding box coordinates in (x, y, w, h) order
        # but we need them in (top, right, bottom, left) order, so we
        # need to do a bit of reordering
        boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

        # compute the facial embeddings for each face bounding box
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []

        # loop over the facial embeddings
        for encoding in encodings:
            # attempt to match each face in the input image to our known
            # encodings
            matches = face_recognition.compare_faces(data["encodings"],
                encoding)
            name = "Unknown" #if face is not recognized, then print Unknown

            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face
                # was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                # determine the recognized face with the largest number
                # of votes (note: in the event of an unlikely tie Python
                # will select first entry in the dictionary)
                name = max(counts, key=counts.get)
                
                #If someone in your dataset is identified, print their name on the screen
                if currentname != name:
                    currentname = name
                    engine.say(currentname)
                    engine.say('On your front ')
                    engine.runAndWait()
                    #   voice code
                    
            # update the list of names
            names.append(name)
            
        # loop over the recognized faces
        for ((top, right, bottom, left), name) in zip(boxes, names):
            # draw the predicted face name on the image – color is in BGR
            cv2.rectangle(frame, (left, top), (right, bottom),
                (0, 255, 0), 2)
            y = top - 15 if top -15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                .8, (255, 0, 0), 2)
        
        # Draw framerate in corner of frame
        cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

        # display the image to our screen
        cv2.imshow("Facial Recognition is Running", frame)
###################################    Mode 2    ##############################
    elif curr_mode == Modes.OCR:
        # Capture frame-by-frame
        frame = videostream.read()
 
        d = pytesseract.image_to_data(frame, output_type=Output.DICT)
        n_boxes = len(d['text'])
        for i in range(n_boxes):
            if int(d['conf'][i]) > 60:
                text = d['text'][i]
                # don't show empty text
                if text and text.strip() != "":
                # store the word to say afterwards
                    engine.say(text+' ')  #* you may delete the space if it is bad voice accent and see
     
        # Display the resulting frame
        cv2.imshow('frame', frame)

        # Generate audio for the text in the frames
        engine.runAndWait()

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1
    
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
videostream.stop()
#GPIO.cleanup()
