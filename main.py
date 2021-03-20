import tensorflow as tf
from PIL import Image
import numpy as np
import keras
from keras.preprocessing.image import load_img, img_to_array

class_names = [
    'Acura',
    'Alfa Romeo',
    'Aston Martin',
    'Audi',
    'Bentley',
    'BMW',
    'Buggati',
    'Buick',
    'Cadiallac'
]

model = tf.keras.models.load_model('C:/Jupyter-Notebooks/carmodel.h5')

model.compile(loss='categorial_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy']
             )

img = Image.open('Cars/imgs_zip/imgs/Audi/Audi_032.jpg')
# img.show()
img = img.resize((224, 224))
img = np.reshape(img, [1, 224, 224, 3])

predictions = model.predict(img)

print(np.argmax(predictions))

clas = class_names[int(np.argmax(predictions)/10000)]
print(clas)
