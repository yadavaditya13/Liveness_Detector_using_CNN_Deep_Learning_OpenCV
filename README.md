# Liveness_Detector_using_CNN_Deep_Learning_OpenCV

This project basically does the task of differentiating between a spoof and real face in live cam. I used Keras which uses Tensorflow as backend to create a CNN and trained the Network using various datasets. Read futher to know about it.

I also used face detector model of OpenCV for detecting faces.

There is a videos folder where you can use your video shot using your web cam or directly from your cells' front or rear camera.
  - There is a script "gathering_my_dataset_from_video.py" that will extract faces from your video frame by frame and store them in respective dataset folder (fake or real...depending on the path and video you provide). Make sure you use your real video and store it in real dir of dataset and vice-versa. 


*** For the spoof video ... simply play your recorded video in front of web-cam and record it or download from youtube or tiktok(atleast useful for something afterall:))... and re-run the previous script to get fake images dataset ***

I also downloaded more dataset from kaggle and did the same for them as well to simply get required ROIs from images and store in dataset dir by creating another script for images named "detecting_faces_from_downloaded_image_dataset.py"

Then in my_script dir you will find My CNN created using Keras... This network is trained using script "model_liveness_train.py"... which yields three output files : 
  - liveness.model (my_self_created_and_trained_model) (p.s:. has an accuracy of about 90%...)
  - le.pickle (our serialized label encoders file)
  - plot (our plotted graph which depicts the results such as losses and accuracy figures of my model)
  
 Then there's the final script "liveness_detection_cam.py" that will use our trained model and detect the classes in real time.
