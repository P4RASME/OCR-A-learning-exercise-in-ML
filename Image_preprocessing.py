import cv2 
import os
from pathlib import Path
import numpy as np 

p = Path(r"C:\Users\ypara\Onedrive\Desktop\OCR_proj\digits")

n_length = 29
dev_tensor_own = []
for f in p.iterdir():
    if f.is_file():
        image = cv2.imread(str(f)) 
        resized_image = cv2.resize(image,(n_length, n_length))
        gray_image = cv2.cvtColor(resized_image,cv2.COLOR_BGR2GRAY)
        array_hopefully = cv2.bitwise_not(gray_image)
        final_array = np.array(array_hopefully)
        dev_tensor_own.append(final_array)

dev_Tensor_own = np.array(dev_tensor_own)
print(dev_Tensor_own)

result_tensor_own = [0,1,2,3,4,5,6,7,8,9]














