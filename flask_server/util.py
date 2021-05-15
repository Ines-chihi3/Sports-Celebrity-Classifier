
#backend = flask server 
#ui is user interface 
#how UI can send image to backend to classify it ?
#1/save image on S3 backet on AWS ans send path for image
#2/send base64 image ?
#base64 incoding is nothing but its a way to convert image to string
#our model use image so we need to convert base64 image to image ?
 
import joblib
import json
import numpy as np
import base64
import cv2
import pywt


#1/load all artifacts we need (model and dictionnary)
#2/tranforme image from base64 to normal image
#3/crop image
#4/use wavelet transform to extract features from image
#5/classify imaeg




#1/load all artifacts we need (model and dictionnary)
#-- private
__class_name_to_number = {}
__class_number_to_name = {}
__model = None
def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __class_name_to_number
    global __class_number_to_name

    with open(r"C:\Users\Ines\Desktop\SportsCelebrityClassification\SportsCelebrityClassification\flask_server\artifacts\class_dictionary.json", "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v:k for k,v in __class_name_to_number.items()}

    global __model
    if __model is None:
        with open(r"C:\Users\Ines\Desktop\SportsCelebrityClassification\SportsCelebrityClassification\flask_server\artifacts\saved_model.pkl", 'rb') as f:
            __model = joblib.load(f)
    print("loading saved artifacts...done")



#2/tranforme image from base64 to normal image
#allow us to get image from base64image send by UI to flask server
def get_cv2_image_from_base64_string(b64str):
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img



#3/crop image
#as we are processing on our model we need to crop image and get image with face and 2 eye
def get_cropped_image_if_2_eyes(image_path, image_base64_data):
    face_cascade = cv2.CascadeClassifier(r"C:\Users\Ines\Desktop\SportsCelebrityClassification\SportsCelebrityClassification\flask_server\opencv\haarcascades\haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(r"C:\Users\Ines\Desktop\SportsCelebrityClassification\SportsCelebrityClassification\flask_server\opencv\haarcascades\haarcascade_eye.xml")

    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)
    #exactly the same code
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cropped_faces = []
    for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) >= 2:
                cropped_faces.append(roi_color)
    return cropped_faces



#4/use wavelet transform to extract features from image
#Now we will use cropped images and apply wavelet transform to extract meaning features that can help with image identification. 
#just like we are processing in model builduiig

def w2d(img, mode='haar', level=1):
    imArray = img
    #Datatype conversions
    #convert to grayscale
    imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    #convert to float
    imArray =  np.float32(imArray)
    imArray /= 255;
    # compute coefficients
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    #Process Coefficients
    coeffs_H=list(coeffs)
    coeffs_H[0] *= 0;

    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode);
    imArray_H *= 255;
    imArray_H =  np.uint8(imArray_H)

    return imArray_H




#5/classify imaeg
#we use the same 
def classify_image(image_base64_data, file_path=None):

    imgs = get_cropped_image_if_2_eyes(file_path, image_base64_data)

    result = []
    for img in imgs:
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))

        len_image_array = 32*32*3 + 32*32
        #we will train this image
        final = combined_img.reshape(1,len_image_array).astype(float)
        result.append({
            #class name
            'class': __class_number_to_name[__model.predict(final)[0]],
            #class prob 
            'class_probability': np.around(__model.predict_proba(final)*100,2).tolist()[0],
            #class number maybe we will need it
            'class_dictionary': __class_name_to_number
        })

    return result

def get_b64_test_image_for_virat():
    with open(r"C:\Users\Ines\Desktop\SportsCelebrityClassification\SportsCelebrityClassification\flask_server\b64.txt") as f:
        return f.read()



if __name__ == '__main__':
    #load artifacts
    load_saved_artifacts()
    #classify_image
    #print(classify_image(get_b64_test_image_for_virat(), None))

    # print(classify_image(None, "./test_images/federer1.jpg"))
    # print(classify_image(None, "./test_images/federer2.jpg"))
    # print(classify_image(None, "./test_images/virat1.jpg"))
    # print(classify_image(None, "./test_images/virat2.jpg"))
    # print(classify_image(None, "./test_images/virat3.jpg")) # Inconsistent result could be due to https://github.com/scikit-learn/scikit-learn/issues/13211
    # print(classify_image(None, "./test_images/serena1.jpg"))
    # print(classify_image(None, "./test_images/serena2.jpg"))
    # print(classify_image(None, "./test_images/sharapova1.jpg"))