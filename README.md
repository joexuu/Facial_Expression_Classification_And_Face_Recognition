# Facial_Expression_Classification_And_Face_Recognition
A CNN-based pytorch implementation on face detection (using retinaface), expression recognition and face recognition, achieving an accuracy of 91.88% on RaDF data

RaDF dateset
The Radboud Faces Database (RaFD) is a set of pictures of 67 models (both adult and children, males and females) displaying 8 emotional expressions.
To download the RaFD dataset, you must request access to the dataset from the Radboud Faces Database website.
****
**demos**  

![final2](https://user-images.githubusercontent.com/43111766/127261061-386c32b9-4cee-4a7c-92c9-00e4cc41f18b.JPG)

****
**Preprocessing RaDF**  
python prepare_data.py

**Train and Eval model**  
python train_emotion_model.py

**Accurary**  
Model：Resnet50 91.88%

**predit the emotion and recognition the face**  
python classify_and_recognition.py



