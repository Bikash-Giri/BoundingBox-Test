# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import AveragePooling2D, Convolution2D, Dropout
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from PIL import Image
import cv2
import random
import matplotlib.pyplot as imshow
from keras.layers import LeakyReLU
from keras import regularizers

from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os

from tensorflow import train



def main():
     #specify the path that contains ur dataset
    data_dir = 'outputs' 

    #check to see if the path is correct 
    os.path.exists(data_dir)

    #read the datasets using pandas libraries
    df = pd.read_csv(os.path.join(data_dir, 'train.csv'))

    print(df.head())

    train_df, test_df = train_test_split(
    df, 
    test_size=0.2,       # 20% test data
    random_state=42,     # for reproducibility
    shuffle=True,        # shuffle before splitting
    stratify=df['PatchLabel']) # optional: keeps label distribution

    train_df.to_csv("train_split.csv", index=False)
    test_df.to_csv("test_split.csv", index=False) 

    # Initialising the lCNN
    classifier = Sequential()

    # Step 1 - Convolution
    classifier.add(Convolution2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu',padding = "same",kernel_regularizer=regularizers.l2(0.001)))

    # Step 2 - Pooling
    classifier.add(AveragePooling2D(pool_size = (2, 2)))
    classifier.add(Dropout(0.2))

    # Step 1 - Convolution
    classifier.add(Convolution2D(16, (3, 3), input_shape = (64, 64, 3), activation = 'relu',padding = "same",kernel_regularizer=regularizers.l2(0.001)))

    # Step 2 - Pooling
    classifier.add(AveragePooling2D(pool_size = (2, 2))) 


    # Step 3 - Flattening
    classifier.add(Flatten())

    # Step 4 - Full connection
    classifier.add(Dense(128, activation = 'relu',kernel_regularizer=regularizers.l2(0.001)))

      # Step 4 - Full connection
    classifier.add(Dense(64, activation = 'relu',kernel_regularizer=regularizers.l2(0.001)))

    classifier.add(Dense(4, activation = 'softmax'))

    # Compiling the CNN
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    """ Create an empty list and read the image and append it to the list which can be used later"""

    temp = []

    for img_name in train_df.Patch:
        patch_path = img_name  
        print(patch_path+ "img_path")

        # img_path = os.path.join(data_dir,"patch",img_name)
        img = cv2.imread(patch_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (64, 64))
        img = img.astype('float32')
        if img is None:
            print(f"Missing: {img_path}")
            continue # this will help us in later stage
        temp.append(img)

    train_x = np.stack(temp)

    temp = []
    for img_name in test_df.Patch:
        img_path = img_name 
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (64, 64))
        if img is None:
            print(f"Missing: {img_path}")
            continue
        temp.append(img.astype('float32'))

    test_x = np.stack(temp)

    train_x = train_x / 255.
    test_x = test_x / 255.

    df = df.reset_index(drop=True)

    # print(train_df.Patch.values)
    #Use label encoder to encode the categorical input

    lb = LabelEncoder()
    train_y = lb.fit_transform(train_df.PatchLabel.values)
    test_y =  to_categorical(train_y)
    type(train_y)

    #Fit our model into a CNN
    #Increase the number of epoch to increase the accurarcy of our CNN as a means of Parameter Tuning
    classifier.fit(train_x, test_y,epochs=20,verbose=1)

    #Generate a random number and read ramdom image from our dataset
    i = random.choice(train_df.index)
   
    img_name = train_df.Patch[i]
    img = cv2.imread(img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32')

    #Final step:Predict our model,check to see the results 
    pred = classifier.predict(train_x)
    pred_classes = np.argmax(pred, axis=1)
    classifier.save("my_model.h5")
    print('Original:', train_df.Patch[i], 'Predicted:', lb.inverse_transform([pred_classes[i]]))
    patch_img = train_df.Patch[i]
    img = cv2.imread(patch_img)
    cv2.imshow("predicted label ",img)
    cv2.waitKey(0)  # waits for key press
    cv2.destroyAllWindows()


if __name__ == '__main__':
   
    main()
