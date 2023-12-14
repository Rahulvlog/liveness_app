# from streamlit_webrtc import webrtc_streamer
# import av
# import cv2
# import tensorflow as tf
# import numpy as np
# import imutils
# import pickle
# import os
# import streamlit as st



# model_path='liveness.model'
# le_path='label_encoder.pickle'
# encodings='encoded_faces.pickle'
# detector_folder='face_detector'
# confidence=0.5
# args = {'model':model_path, 'le':le_path, 'detector':detector_folder, 
# 	'encodings':encodings, 'confidence':confidence}

# # load the encoded faces and names
# print('[INFO] loading encodings...')
# with open(args['encodings'], 'rb') as file:
# 	encoded_data = pickle.loads(file.read())
 
 
 

# # load our serialized face detector from disk
# print('[INFO] loading face detector...')
# proto_path = os.path.sep.join([args['detector'], 'deploy.prototxt'])
# model_path = os.path.sep.join([args['detector'], 'res10_300x300_ssd_iter_140000.caffemodel'])
# detector_net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
	
# # load the liveness detector model and label encoder from disk
# liveness_model = tf.keras.models.load_model(args['model'])
# le = pickle.loads(open(args['le'], 'rb').read())

# print("start this code")
# class VideoProcessor:		
# 	def recv(self, frame):
        
# 		frm = frame.to_ndarray(format="bgr24")

# 		# iterate over the frames from the video stream
# 		# while True:
# 			# grab the frame from the threaded video stream
# 			# and resize it to have a maximum width of 600 pixels
# 		frm = imutils.resize(frm, width=800)

# 		# grab the frame dimensions and convert it to a blob
# 		# blob is used to preprocess image to be easy to read for NN
# 		# basically, it does mean subtraction and scaling
# 		# (104.0, 177.0, 123.0) is the mean of image in FaceNet
# 		(h, w) = frm.shape[:2]
# 		blob = cv2.dnn.blobFromImage(cv2.resize(frm, (300,300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
		
# 		# pass the blob through the network 
# 		# and obtain the detections and predictions
# 		detector_net.setInput(blob)
# 		detections = detector_net.forward()
		
# 		# iterate over the detections
# 		for i in range(0, detections.shape[2]):

# 			# extract the confidence (i.e. probability) associated with the prediction
# 			confidence = detections[0, 0, i, 2]
            
			
# 			# filter out weak detections
# 			if confidence > args['confidence']:
# 				# compute the (x,y) coordinates of the bounding box
# 				# for the face and extract the face ROI
# 				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
# 				(startX, startY, endX, endY) = box.astype('int')
				
# 				# expand the bounding box a bit
# 				# (from experiment, the model works better this way)
# 				# and ensure that the bounding box does not fall outside of the frame
# 				startX = max(0, startX-20)
# 				startY = max(0, startY-20)
# 				endX = min(w, endX+20)
# 				endY = min(h, endY+20)
				
# 				# extract the face ROI and then preprocess it
# 				# in the same manner as our training data

# 				face = frm[startY:endY, startX:endX] # for liveness detection
# 				# expand the bounding box so that the model can recog easier

# 				# some error occur here if my face is out of frame and comeback in the frame
# 				try:
# 					face = cv2.resize(face, (32,32)) # our liveness model expect 32x32 input
# 				except:
# 					break

# 				# initialize the default name if it doesn't found a face for detected faces
# 				name = 'Unknown'
# 				face = face.astype('float') / 255.0 
# 				face = tf.keras.preprocessing.image.img_to_array(face)

# 				# tf model require batch of data to feed in
# 				# so if we need only one image at a time, we have to add one more dimension
# 				# in this case it's the same with [face]
# 				face = np.expand_dims(face, axis=0)
			
# 				# pass the face ROI through the trained liveness detection model
# 				# to determine if the face is 'real' or 'fake'
# 				# predict return 2 value for each example (because in the model we have 2 output classes)
# 				# the first value stores the prob of being real, the second value stores the prob of being fake
# 				# so argmax will pick the one with highest prob
# 				# we care only first output (since we have only 1 input)
# 				preds = liveness_model.predict(face)[0]
# 				j = np.argmax(preds)
# 				label_name = le.classes_[j] # get label of predicted class
				
# 				# draw the label and bounding box on the frame
# 				label = f'{label_name}: {preds[j]:.4f}'
# 				print(f'[INFO] {name}, {label_name}')
				
# 				if label_name == 'fake':
# 					cv2.putText(frm, "Fake Alert!", (startX, endY + 25), 
# 								cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,255), 2)
				
# 				cv2.putText(frm, name, (startX, startY - 35), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,130,255),2 )
# 				cv2.putText(frm, label, (startX, startY - 10),
# 							cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
# 				cv2.rectangle(frm, (startX, startY), (endX, endY), (0, 0, 255), 4)

# 		return av.VideoFrame.from_ndarray(frm, format='bgr24')

# webrtc_streamer(key="key", video_processor_factory=VideoProcessor,rtc_configuration={
# 		"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
# 	},sendback_audio=False, video_receiver_size=1)


from streamlit_webrtc import webrtc_streamer
import av
import cv2
import tensorflow as tf
import numpy as np
import imutils
import pickle
import os
import streamlit as st
from skimage.metrics import structural_similarity as ssim

model_path = 'liveness.model'
le_path = 'label_encoder.pickle'
encodings = 'encoded_faces.pickle'
detector_folder = 'face_detector'
confidence = 0.5
args = {'model': model_path, 'le': le_path, 'detector': detector_folder,
        'encodings': encodings, 'confidence': confidence}

# load the encoded faces and names
print('[INFO] loading encodings...')
with open(args['encodings'], 'rb') as file:
    encoded_data = pickle.loads(file.read())

# load our serialized face detector from disk
print('[INFO] loading face detector...')
proto_path = os.path.sep.join([args['detector'], 'deploy.prototxt'])
model_path = os.path.sep.join([args['detector'], 'res10_300x300_ssd_iter_140000.caffemodel'])
detector_net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

# load the liveness detector model and label encoder from disk
liveness_model = tf.keras.models.load_model(args['model'])
le = pickle.loads(open(args['le'], 'rb').read())

print("start this code")


class VideoProcessor:
    
    def __init__(self):
        # Load the hardcoded image for comparison
        self.hardcoded_image = cv2.imread('test_pics/random.jpg')  # Provide the actual path

    
    def recv(self, frame):

        frm = frame.to_ndarray(format="bgr24")

        # iterate over the frames from the video stream
        # while True:
        # grab the frame from the threaded video stream
        # and resize it to have a maximum width of 600 pixels
        frm = imutils.resize(frm, width=800)

        # grab the frame dimensions and convert it to a blob
        # blob is used to preprocess image to be easy to read for NN
        # basically, it does mean subtraction and scaling
        # (104.0, 177.0, 123.0) is the mean of image in FaceNet
        (h, w) = frm.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frm, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        # pass the blob through the network
        # and obtain the detections and predictions
        detector_net.setInput(blob)
        detections = detector_net.forward()

        # iterate over the detections
        for i in range(0, detections.shape[2]):
            

            # extract the confidence (i.e. probability) associated with the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections
            if confidence > args['confidence']:
                # compute the (x,y) coordinates of the bounding box
                # for the face and extract the face ROI
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype('int')

                # expand the bounding box a bit
                # (from experiment, the model works better this way)
                # and ensure that the bounding box does not fall outside of the frame
                startX = max(0, startX - 20)
                startY = max(0, startY - 20)
                endX = min(w, endX + 20)
                endY = min(h, endY + 20)

                # extract the face ROI and then preprocess it
                # in the same manner as our training data

                face = frm[startY:endY, startX:endX]  # for liveness detection
                # expand the bounding box so that the model can recognize easier
                

                # some error occur here if my face is out of frame and comeback in the frame
                try:
                    face = cv2.resize(face, (32, 32))  # our liveness model expects 32x32 input
                except:
                    break
                
                match_percentage = self.match_with_hardcoded_image(face)
                print("this is the match percentage: ", match_percentage)

                # initialize the default name if it doesn't find a face for detected faces
                name = 'known'
                face = face.astype('float') / 255.0
                face = tf.keras.preprocessing.image.img_to_array(face)

                # tf model requires a batch of data to feed in
                # so if we need only one image at a time, we have to add one more dimension
                # in this case it's the same with [face]
                face = np.expand_dims(face, axis=0)

                # pass the face ROI through the trained liveness detection model
                # to determine if the face is 'real' or 'fake'
                # predict return 2 value for each example (because in the model we have 2 output classes)
                # the first value stores the prob of being real, the second value stores the prob of being fake
                # so argmax will pick the one with the highest prob
                # we care only first output (since we have only 1 input)
                preds = liveness_model.predict(face)[0]
                # print("predicion", preds)
                j = np.argmax(preds)
                # print("predicion", j)
                # print("le the", le.classes_)
                label_name = le.classes_[j]  # get label of predicted class
                # print("predicion", label_name)
                
                

                # draw the label and bounding box on the frame
                label = f'{label_name}: {preds[j]:.4f}'
                # print(f'[INFO] {name}, {label_name}')
                if preds[j] > 0.8 and match_percentage > 0.5:
                    cv2.putText(frm, name, (startX, startY - 35), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 130, 255), 2)
                    cv2.putText(frm, "Real", (startX, startY - 10),
                                cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
                    # if label_name == 'fake':
                    
                else:
                    cv2.putText(frm, "Fake Alert!", (startX, endY + 25),
                                cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frm, "Fake", (startX, startY - 35), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 130, 255), 2)
                    

                cv2.rectangle(frm, (startX, startY), (endX, endY), (0, 0, 255), 4)


        return av.VideoFrame.from_ndarray(frm, format='bgr24')
    
    
    def match_with_hardcoded_image(self, face):
        face = cv2.resize(face, (self.hardcoded_image.shape[1], self.hardcoded_image.shape[0]))
        
        # Convert images to grayscale for structural similarity index
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        hardcoded_image_gray = cv2.cvtColor(self.hardcoded_image, cv2.COLOR_BGR2GRAY)
        print("face_gray, ", face_gray)
        print("hardcoded_image_gray", hardcoded_image_gray)

        # Calculate the Structural Similarity Index (SSI)
        ssi_index, _ = ssim(face_gray, hardcoded_image_gray, full=True)
        match_percentage = (ssi_index + 1) * 50
        
        
        # Return the match ratio
        return match_percentage
    

webrtc_streamer(key="key", video_processor_factory=VideoProcessor, rtc_configuration={
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}, sendback_audio=False, video_receiver_size=1)

