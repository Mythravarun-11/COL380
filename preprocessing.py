import cv2,os
import numpy as np

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (28, 28))
    gray = gray / 255
    filename = os.path.basename(image_path)
    output_file_path = "./pre-proc-img/"+filename[:-4]+".txt"
    
    np.savetxt(output_file_path, gray, fmt='%1.6f')
    return

if __name__ == "__main__":
    input_folder = './img/'

    for image in os.listdir(input_folder):
        preprocess_image(input_folder+image)
       
    
