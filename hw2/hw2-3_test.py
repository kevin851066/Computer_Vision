
# coding: utf-8

# In[8]:


"""
• python3 hw2-3_test.py $1 $2
• $1: directory of the testing images folder
• $2: path of the output prediction file
• E.g., python3 hw2-3_test.py ./test_images/ ./output.csv
• Testing images folder include images named:
• 0000.png , 0002.png , ... , 9999.png
• Output prediction file format • In csv format
• First row: “id,label”
• From second row: “<image_id>, <predicted_label>”
"""

import cv2
import os 
import sys
import numpy as np
from keras.models import load_model
from keras.utils import to_categorical

model = load_model('model/model.lenet_5_new')

test_img = []
    
for j in range(10000):
    j = "%04d" % j
    file_name = str(j) + '.png'
    image = cv2.imread(os.path.join(sys.argv[1], file_name), 0)
    test_img.append(image)
    
test_img = np.array(test_img)
test_img = np.expand_dims(test_img, axis = -1) / 255

    


# In[14]:


pres = model.predict(test_img)
y_classes = pres.argmax(axis=-1)
# print(y_classes.shape)
# print(y_classes)

import csv

# 開啟輸出的 CSV 檔案
with open(sys.argv[2], 'w', newline='') as csvfile:
  # 建立 CSV 檔寫入器
    writer = csv.writer(csvfile)

  # 寫入一列資料
    writer.writerow(['id', 'label'])
    for j in range(10000):
        k = j
        j = "%04d" % j
        writer.writerow([j, y_classes[k]])

print("write in successfully!")
    


# In[13]:



