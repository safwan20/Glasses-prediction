# Glasses-prediction

# About dataset  
The dataset has been taken from this website https://www.kaggle.com/jeffheaton/glasses-or-no-glasses?select=faces-spring-2020.

# About model architecture  
layers : Conv2D--->MaxPool2D---->Conv2D--->MaxPool2D---->Conv2D--->MaxPool2D---->Flatten--->Dense---->Dropout--->Dense  
activation : (relu)              (relu)                  (relu)                            (relu)                (sigmoid)  
fliter : 32,(3,3)  
pool_size = (2,2)  
input_shape = (48,48,1)  
output_shape = 128  

# Metrics  
loss='binary_crossentropy'  
optimizer='adam'  
metrics=['accuracy']  

# Save architecture  
I have used h5 format for saving.

#Used the trained model glass.h5  
from keras.models import load_model  
model = load_model('glass.h5')  
model.predict_classes(image)  
