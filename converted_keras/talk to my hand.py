#Whenever the subject will raise hand or hand is observed on webcam in builtled in Arduino will glow
#before running the python programm go to Arduino IDE->files->examples->Firmata->standardFirmata and run the code in the device
#Also note the port on arduino
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import cv2
from pyfirmata import Arduino, util 
import time


# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# In order to convert cv2 image into pillow for processing we had to define cv2_to_pil
def cv2_to_pil(frame): 
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#defining port on which arduino is used PUT THE PORT INFORMATION HERE BELOW
board = Arduino('/dev/cu.usbmodem14201')

#defining the camera since only webcam is present in system hence the value 0
cam = cv2.VideoCapture(0)

#Image processing initiates
while True:
    ret, frame = cam.read()     #defining frame   
    image = cv2_to_pil(frame)   #conversion of cv2->pil for further processing by tf

    
    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)

    # display the resized image
    #image.show()               #only activate when processing image otherwise screen will be filled by frame shots of the video

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    
    #since prediction is an 2d array having floats it had to be converted into strings for representation on the screen
    #the prediction has 2 values containing models certaininty of both the classes {at [0,0]->hand : at [0,1]-> not hand
    init1=int(prediction[0,0]*100) #taking [0,0] of array and converting it into int
    hand='certain='+str(init1)+'%' #conversion of int into strings
    init2=int(prediction[0,1]*100)
    not_hand='uncertain='+str(init2)+'%'
    
    #putText function is used to display text on the video {frame->name of image: hand-> string to be displayed: 
    frame = cv2.putText(frame,hand,(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2) #cv2.FONT_HERSHEY_SIMPLEX-> font: 2->thickness:
    frame = cv2.putText(frame,not_hand,(800,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2) #(b,g,r)->colour

    #displaying the processed image as video output
    cv2.imshow("Talk to my hand", frame)

    #print('hand=',prediction[0,0])     #activate to get real data from prediction
    #print('no hand=',prediction[0,1])

    #use to send the command to arduino
    if(prediction[0,0]>=float(0.8)):
        board.digital[13].write(1)
        time.sleep(0.1)
    else:
        board.digital[13].write(0)
        time.sleep(0.1)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
