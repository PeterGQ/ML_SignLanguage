import base64
import io
import sys
import tempfile
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from PIL import Image, ImageOps
import numpy as np
import math
import tensorflow as tf
import streamlit as stre
from io import BytesIO,StringIO

# Followed this video tutorial
# https://www.youtube.com/watch?v=wa2ARoUUdU8
# We Train the model in Teachable machine

# tmp = tempfile.NamedTemporaryFile(suffix=".jpg")

STYLE = """
<style>
img{
    max-width: 100%
}
</style>
"""
def SignLang(file):
    offset = 20
    imgSize = 300
    labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U",
              "V", "W", "X", "Y", "Z"]
    k = cv2.waitKey(0)
    detector = HandDetector(maxHands=1)
    classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

    img = cv2.imread(file)
    if img is None:
        stre.write("no img")
        sys.exit("Image not found")
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    # Detects hand pressence
    if hands:
        try:
            # B/C the size of hand is modular, we need a set size param to fit it in
            # Thus the implmentation of imgWhite
            # Detects the x, y, width, and height
            hand = hands[0]
            x, y, w, h = hand['bbox']

            # Creates a white background
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            # Overlay the White on Hand
            imgCropShape = imgCrop.shape

            # We need to stretch the boundries such that we leave no void space from overlay
            aspectRatio = h / w
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape

                # We need to center it, we have a set width
                # so we need to shift w to be 300
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize

                # Send our image to our model, it well classify img
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                print(prediction, index)

            else:
                k = imgSize / w
                hCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape

                # We need to center it, we have a set height
                # so we need to shift h to be 300
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                # Send our image to our model, it well classify img
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                print(prediction, index)

            cv2.putText(imgOutput, labels[index], (x, y - offset), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2, )
            cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + -offset), (255, 0, 255), 2)
        except:
            print("Hands out of Bounds Error")
        # cv2.imshow("Image", imgOutput)
        # key = cv2.waitKey(0)
    else:
        print("No hands detected Bozo")
    return labels[index]

def Website():
    # stre.info(__doc__)
    stre.markdown(STYLE, unsafe_allow_html=True)

    with stre.sidebar:
        stre.markdown(f'<h1 style="text-align: center">Collaboraters</h1>', unsafe_allow_html=True)

        stre.markdown('----')

        stre.markdown(f'<h2 style="text-align: left">Alfonso Martinez:</h2>', unsafe_allow_html=True)
        '''
        Junior in Computer Science - [![Repo](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/OwlFunsOh/Personal) 
        '''
        stre.markdown(f'<h2 style="text-align: left">Sean Belon:</h2>', unsafe_allow_html=True)
        '''
        Junior in Computer Science - [![Repo](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/Dream-Yami) 
        '''
        stre.markdown(f'<h2 style="text-align: left">Darsh Patel:</h2>', unsafe_allow_html=True)
        '''
        Sophmore in Computer Science - [![Repo](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/darshp623/Personal-Projects) 
        '''
        stre.markdown(f'<h2 style="text-align: left">Peter Gatira:</h2>', unsafe_allow_html=True)
        '''
        Junior in Computer Science - [![Repo](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/PeterGQ) 
        '''
        stre.markdown('----')

    """
    # American Sign Language Recognition
    """
    file1 = open("./ASL.gif","rb")
    contents = file1.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file1.close()
    stre.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="ASL gif">', unsafe_allow_html=True)
    stre.text("")
    stre.text("")
    stre.text("")
    """
        # The Problem
    """
    stre.subheader("I wish I could practice sign language \U0001F614")
    stre.write("Have you ever wanted to practice sign language, but you had no one to practice it with? Fear not, using our model, "
               "you can have Artificial Intelligence help you practice sign language to your heart’s content! In our project ")
    """
        # The Goal
    """
    stre.write("The aim of our project is to take in any hand sign, and through AI and Machine Learning, have the image be interpreted to determine what letter in the alphabet it is. "
               "Not only does this create a useful and helpful experience that many can use to learn American Sign Language, but it does so in a fun,"
               " easy way which can be accessed by any individual, of any age, regardless of your prior experience with American Sign Language.")
    """
        # The Solution
    """
    stre.write("There were two stages to get the project completed. The first stage required making a function that would take a picture, "
               "feed the picture into a model, and take the model’s output and return it to the user. The second stage of the project was to create "
               "a program that would take live data from our webcam so that we can take pictures of American Sign Language hand signals. Those hand signals would then "
               "be given as a dataset to the module in order to train it to recognize the different letters.")
    stre.subheader("Stage 1:")
    stre.write("The code is one function that takes in an image, and outputs what the machine thinks is the most probable letter the image represents. First, the program "
               "checks if an image input was not empty. If it is not empty, then we use a Hand tracking Module to help detect hands within the picture. If a Hand is detected, "
               "it tracks the position of the hand. After the tracker finds the hand, the program crops a rectangle around the hand so that the cropped image would contain the "
               "hand with very minimal background space. The image is then fed into the “classifier” which is a variable that contains the file to the trained module. The trained"
               " module would make a prediction on what the image hand signal represents, and returns an index. The program finally returns the alphabet character at the specified"
               " position that the module returned.")
    stre.subheader("Stage 2:")
    stre.write("In order for the module to work, it needs to be fed a dataset. Instead of finding many hand signal pictures on the internet, it was easier and more time efficient to"
               " create our own dataset. To do this, a similar process is done as stage 1. First, we needed to create a variable that allows our webcam to continuously record. Once the"
               " video from the webcam is running, we use the Hand Tracking Module to track our hand in real time. Next, we had to make our image cleaner so that our dataset would have some"
               " consistent sizes. First, we made an overlay so that the video was bounded by a 300 x 300 pixel box. Then, we centered the image within the box. Finally, we calculated an equation"
               " to resize the video image so that the video of the hand would always take up all of the space in the 300 x 300 pixel box. Once this was done, we set up a key that would be used to tak"
               "e pictures of the video, and save it to a designated location. To gather enough data, approximately 300 pictures of each sign language symbol that represents an alphabetical character was"
               " used. In total that would be around 7,800 pictures. Within each symbol, we made sure that there were slight deviations from the “perfect” hand signal such as hand tilt or different backgrounds."
               " This was to ensure that the module would be able to recognize such deviations, and apply its “knowledge” on a more general level. The more general the module is trained to identify hand signals, "
               " more accurate it will be when it is tested by other people’s hand signals. ")
    file = stre.file_uploader("Upload Images", type=["csv","png","jpg"])
    stre.markdown('----')
    show_file = stre.empty()
    if isinstance(file, BytesIO):
        show_file.image(file)
    stre.markdown('----')
    if file is not None:
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", dir='.')
        img = Image.open(io.BytesIO(file.read()))
        # img_path = "./{}".format(file)
        img.save(tmp)
        img_path = "./{}".format(tmp)
        stre.write(tmp.name)
        predictedVal = SignLang(tmp.name)
        tmp.close()
        stre.markdown(f'<h1 style="text-align: center">predicted value</h1>', unsafe_allow_html=True)
        stre.markdown(f'<h3 style="text-align: center">Letter: {predictedVal}</h3>', unsafe_allow_html=True)

if __name__ == '__main__':
    Website()