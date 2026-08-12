import cv2 
import os
from pathlib import Path


p = Path(r"C:\Users\ypara\Onedrive\Desktop\OCR_proj\digits")

n_length = 100
for f in p.iterdir():
    if f.is_file():
        image = cv2.imread(str(f)) 
        resized_image = cv2.resize(image,(n_length, n_length))
        final_image = cv2.convertScaleAbs(resized_image,alpha = 5, beta = -200)
        cv2.imwrite(f"C:\\Users\\ypara\\OneDrive\\Desktop\\OCR_proj\\digits_processed\\{f.stem}_processed.png", final_image)
        print("one digit done")
















