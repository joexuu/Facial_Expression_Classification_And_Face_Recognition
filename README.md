# Facial_Expression_Classification_And_Face_Recognition
A CNN-based pytorch implementation on face detection (using retinaface), expression recognition and face recognition, achieving an accuracy of 91.88% on RaDF data

RaDF dateset
The Radboud Faces Database (RaFD) is a set of pictures of 67 models (both adult and children, males and females) displaying 8 emotional expressions.
To download the RaFD dataset, you must request access to the dataset from the Radboud Faces Database website.

Preprocessing RaDF
python prepare_data.py

Train and Eval model
python train_emotion_model.py

Accurary
Model：Resnet50 91.88%

predit the emotion and recognition the face
python classify_and_recognition.py

Demos
![2](https://user-images.githubusercontent.com/43111766/127254265-5494829f-6c82-44d4-b739-20c8cf76e463.JPG)
![20150212_7479](https://user-images.githubusercontent.com/43111766/127254346-d5fd3a1c-6ddf-48f1-8f19-3d154270b4b6.jpg)
![3](https://user-images.githubusercontent.com/43111766/127254356-76eedeed-46a7-4971-a44c-6de0e098eadb.JPG)
