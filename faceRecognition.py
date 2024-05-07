from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import math
import numpy as np
import argparse
import imutils
import time
import dlib
import os
# import RPi.GPIO as GPIO
import pygame.mixer
from tkinter import *
import cv2
from datetime import datetime





class FaceRecognition(Toplevel):
    def __init__(self):
       
        

        
        def eye_aspect_ratio(eye):
            # compute the euclidean distances between the two sets of
            # vertical eye landmarks (x, y)-coordinates
            A = dist.euclidean(eye[1], eye[5])
            B = dist.euclidean(eye[2], eye[4])

            # compute the euclidean distance between the horizontal
            # eye landmark (x, y)-coordinates
            C = dist.euclidean(eye[0], eye[3])

            # compute the eye aspect ratio
            ear = (A + B) / (2.0 * C)

            # return the eye aspect ratio
            return ear
         
        # construct the argument parse and parse the arguments
         
        # define two constants, one for the eye aspect ratio to indicate
        # blink and then a second constant for the number of consecutive
        # frames the eye must be below the threshold
        EYE_AR_THRESH = 0.2
        EYE_AR_CONSEC_FRAMES = 2

        # initialize the frame counters and the total number of blinks
        COUNTER = 0
        TOTAL = 0
        vs = VideoStream(0).start()
        # initialize dlib's face detector (HOG-based) and then create
        # the facial landmark predictor
        print("[INFO] loading facial landmark predictor...")
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        # grab the indexes of the facial landmarks for the left and
        # right eye, respectively
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        # start the video stream thread
        print("[INFO] starting video stream thread...")




        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('trainer/trainer.yml')
        cascadePath = "haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(cascadePath);
        #gun_cascade
        gun_cascade = None
        try:
            gun_cascade = cv2.CascadeClassifier('knifecascade.xml')
        except cv2.error as e:
            print("OpenCV Error:", e)
        first_frame = None
        gun_exists = None

        font = cv2.FONT_HERSHEY_SIMPLEX

        #iniciate id counter
        id1 = 0

        # names related to ids: example ==> Marcelo: id=1,  etc
         
        #vs = VideoStream(usePiCamera=True).start()


        # Define min window size to be recognized as a face




        

        # loop over frames from the video stream
        while True:
            try:
            # if this is a file video stream, then we need to check if
            # there any more frames left in the buffer to process
            
            # grab the frame from the threaded video file stream, resize
            # it, and convert it to grayscale
            # channels)
            
            #frame = vs.read()
            #
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                img =vs.read()
                img = cv2.flip(img, 1) # Flip vertically
                img = imutils.resize(img, width=550,height=250)
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                
                
                knives = gun_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=10, minSize=(150, 150))
                
                
                # detect faces in the grayscale frame
                
                
                faces = faceCascade.detectMultiScale( 
                    gray,
                    scaleFactor = 1.2,
                    minNeighbors = 5,
                    minSize = ( 45, 20),
                )
                
                rects = detector(gray, 0)
                
                # loop over the face detections
                for rect in rects:
                    # determine the facial landmarks for the face region, then
                    # convert the facial landmark (x, y)-coordinates to a NumPy
                    # array
                    
                    if len(knives) > 0:
                        gun_exists = True

                    for (x,y,w,h) in knives:
                        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
                        

                    if first_frame is None:
                        first_frame = gray
                        continue
                   
                    key = cv2.waitKey(1) & 0XFF
                    if key == ord('q'):
                        break    
                    
                    if gun_exists:
                        print('Gun exists')
                    else:
                        print('Gun not found')
                    
                    
                          
                    shape = predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)

                    # extract the left and right eye coordinates, then use the
                    # coordinates to compute the eye aspect ratio for both eyes
                    leftEye = shape[lStart:lEnd]
                    rightEye = shape[rStart:rEnd]
                    leftEAR = eye_aspect_ratio(leftEye)
                    rightEAR = eye_aspect_ratio(rightEye)

                    # average the eye aspect ratio together for both eyes
                    ear = (leftEAR + rightEAR) / 2.0

                    # compute the convex hull for the left and right eye, then
                    # visualize each of the eyes
                    leftEyeHull = cv2.convexHull(leftEye)
                    rightEyeHull = cv2.convexHull(rightEye)
                    cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
                    cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)

                    # check to see if the eye aspect ratio is below the blink
                    # threshold, and if so, increment the blink frame counter
                    if ear < EYE_AR_THRESH:
                        COUNTER += 1
                        for(x,y,w,h) in faces:
                            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
                            id1, confidence = recognizer.predict(gray[y:y+h,x:x+w])
                            confidence = int(confidence)
                            if (confidence < 90): 
                                id="Family Member"
                                print(confidence)
                                # GPIO.setmode(GPIO.BCM)
                                # GPIO.setup(26, GPIO.OUT)
                                try:
                                    print("set GIOP high")
                                    #  GPIO.output(26, True)
                                    # time.sleep(1)
                                    #  GPIO.output(26,False)
                                    # time.sleep(1)
                                except KeyboardInterrupt: # If CTRL+C is pressed, exit cleanly:
                                    print("Keyboard interrupt")

                                except:
                                 print("some error") 

                                finally:
                                    print("clean up")
                                #    GPIO.cleanup()
                                
                                
                            
                            
                               
                                
                            else:
                                id = "unknown"
                                pygame.mixer.init()
                                pygame.mixer.music.load("doorbell.mp3")
                                pygame.mixer.music.play(2)
                               
                                print(confidence)
                        
                            
                           
                            confidence_str = "{:.2f}%".format(confidence)
                            cv2.putText(img, str(id), (x+5, y-5), font, 1, (255, 255, 255), 2)
                            cv2.putText(img, str(confidence_str), (x+5, y+h-5), font, 1, (255, 255, 0), 1)

                    
                    # otherwise, the eye aspect ratio is not below the blink
                    # threshold
                    else:
                        # if the eyes were closed for a sufficient number of
                        # then increment the total number of blinks
                        if COUNTER >= EYE_AR_CONSEC_FRAMES:
                            TOTAL += 1

                        # reset the eye frame counter
                        COUNTER = 0

                    # draw the total number of blinks on the frame along with
                    # the computed eye aspect ratio for the frame
                # function that handles the trackbar
            
                # show the frame
                
                
                cv2.imshow("ADUAS", img)
                
                cv2.waitKey(1)  # Add a small delay
                # Press 'ESC' for exiting video
            
            
                # if the `x` key was pressed, break from the loop
                # if cv2.waitKey(1) & 0xFF == ord('q') or (cv2.getWindowProperty('ADUAS', 1)) == -1.0:
                #     # print("Breaking from the loop")
                #     break
                    
                    
            except Exception as e:
                print(f"An error occurred: {e}")
                break        
            

        # do a bit of cleanup
        # do a bit of cleanup
        print("Cleaning up...")
        vs.stop()
        cv2.destroyAllWindows()
        print("Window property after the loop:", cv2.getWindowProperty('ADUAS', 1))
        print("Cleanup complete")




