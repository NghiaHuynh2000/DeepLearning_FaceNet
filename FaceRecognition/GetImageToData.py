import numpy as np
import cv2
import numpy as np
import sqlite3
import mysql.connector
import os
import tensorflow.compat.v1 as tf
import facenet
import detect_face

import pickle
from PIL import Image

import time

video= 0
modeldir = './model/20180402-114759.pb'
classifier_filename = './class/classifier.pkl'
npy='./npy'
train_img="./train_img"

def insertOrupdate(id,name):
    myconn = mysql.connector.connect(host = "localhost", user = "root", passwd = "nhattaisin9999", database = "nhan_dien_sv")
    query = "SELECT * FROM people WHERE id="+str(id)

    cur = myconn.cursor()
    cur.execute(query)
    records = cur.fetchall()
    
    isRecordExist=0
    for row in records:
        isRecordExist=1
    if(isRecordExist==0):
        query="INSERT INTO people(ID,Name) VALUES("+str(id)+",'"+str(name)+"')"
    else:
        query="UPDATE people SET Name='"+str(name)+"' Where id="+str(id)
    cur.execute(query)
    myconn.commit()
    cur.close()
    myconn.close()

def Get_Image_To_Database(id,name):

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)
            minsize = 30  # minimum size of face
            threshold = [0.7,0.8,0.8]  # three steps's threshold
            factor = 0.709  # scale factor
            margin = 44
            batch_size =100 #1000
            image_size = 182
            input_image_size = 160
            facenet.load_model(modeldir)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            classifier_filename_exp = os.path.expanduser(classifier_filename)
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile,encoding='latin1')

            cap = cv2.VideoCapture(0)
            #insert to db:
            insertOrupdate(id,name)
            sampleNum=0
            while (True):
                ret, frame = cap.read()

                if not os.path.exists('train_img'):
                        os.makedirs('train_img')
                if not os.path.exists('train_img/'+str(id)):
                        os.makedirs('train_img/'+str(id))
                sampleNum=sampleNum+1
                cv2.imwrite('train_img/'+str(id)+"/"+str(sampleNum)+".jpg",frame) 

                timer =time.time()
                if frame.ndim == 2:
                    frame = facenet.to_rgb(frame)
                bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                faceNum = bounding_boxes.shape[0]
                if faceNum > 0:
                            det = bounding_boxes[:, 0:4]
                            img_size = np.asarray(frame.shape)[0:2]
                            cropped = []
                            scaled = []
                            scaled_reshape = []
                            for i in range(faceNum):
                                emb_array = np.zeros((1, embedding_size))
                                xmin = int(det[i][0])
                                ymin = int(det[i][1])
                                xmax = int(det[i][2])
                                ymax = int(det[i][3])
                                try:
                                    # inner exception
                                    if xmin <= 0 or ymin <= 0 or xmax >= len(frame[0]) or ymax >= len(frame):
                                        print('Face is very close!')
                                        continue
                                    cropped.append(frame[ymin:ymax, xmin:xmax,:])
                                    cropped[i] = facenet.flip(cropped[i], False)
                                    scaled.append(np.array(Image.fromarray(cropped[i]).resize((image_size, image_size))))
                                    scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),
                                                            interpolation=cv2.INTER_CUBIC)
                                    scaled[i] = facenet.prewhiten(scaled[i])
                                    scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
                                    feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                                    emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                                    predictions = model.predict_proba(emb_array)
                                    best_class_indices = np.argmax(predictions, axis=1)
                                    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                                    if best_class_probabilities>0.87:
                                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)    #boxing face
                                except:   
                        
                                    print("error")
             #   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
             #   faces = face_cascade.detectMultiScale(gray)
             #   for (x, y, w, h) in faces:
             #       cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
             #       if not os.path.exists('dataset'):
             #           os.makedirs('dataset')
             #       sampleNum=sampleNum+1
             #       cv2.imwrite('dataset/User.'+str(id)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w]) 

             #       roi_gray = gray[y:y + h, x:x + w]
             #       roi_color = frame[y:y + h, x:x + w]
                endtimer = time.time()
                fps = 1/(endtimer-timer)

              
                roi_color = frame;
                cv2.imshow('Chụp ảnh', frame)
                cv2.waitKey(1)
                if(sampleNum==500):
                    break
            cap.release()
            cv2.destroyAllWindows()


