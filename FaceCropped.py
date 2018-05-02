import cv2
import os

def facecrop(image):
    facedata = "C:/Users/karth/Documents/ML/Project/haarcascade_frontalface_alt.xml"
    cascade = cv2.CascadeClassifier(facedata)

    img = cv2.imread(image)

    minisize = (img.shape[1],img.shape[0])
    miniframe = cv2.resize(img, minisize)

    faces = cascade.detectMultiScale(miniframe)
    path1 = "C:/Users/karth/Documents/ML/Project/Dataset/Cropped_train/Female/"
    for f in faces:
        x, y, w, h = [ v for v in f ]
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255))

        sub_face = img[y:y+h, x:x+w]
        fname, ext = os.path.splitext(image)
        fname = os.path.basename(image)
        cv2.imwrite(path1+"Cropped_"+fname,sub_face)
            

    return


for filename in os.listdir("C:/Users/karth/Documents/ML/Project/Dataset/Training/Female"):
    print(filename)
    facecrop("C:/Users/karth/Documents/ML/Project/Dataset/Training/Female/"+filename)