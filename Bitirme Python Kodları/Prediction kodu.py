from keras.layers import Dense, Dropout, Conv3D, Input, MaxPool3D, Flatten, Activation, ZeroPadding3D, AveragePooling3D
from keras.regularizers import l2
from keras.layers import Dense, Dropout, Conv3D, Input, MaxPool3D, Flatten, Activation
from keras.regularizers import l2
from keras.models import Model
import random
import numpy as np
import cv2
import os
from keras.optimizers import SGD, Adam, Adagrad
from keras.utils import np_utils
import matplotlib.pyplot as plt


def c3d_model():
    input_shape = (112, 112, 20, 3)
    weight_decay = 0.005
    nb_classes = 101

    inputs = Input(input_shape)
    x = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu', kernel_regularizer=l2(weight_decay))(inputs)
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x = Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x = Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)


    x = Flatten()(x)
    x = Dense(512, input_dim=4096, kernel_initializer='glorot_normal', kernel_regularizer=l2(0.001), activation='relu')(x)
    x = Dropout(0.6)(x)
    x = Dense(32, kernel_initializer='glorot_normal', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.6)(x)
    x = Dense(6, kernel_initializer='glorot_normal', kernel_regularizer=l2(0.001), activation='softmax')(x)

    model = Model(inputs, x)
    return model


f = open('class_names.txt', 'r')
class_names = f.readlines()
f.close()

# init model
model = c3d_model()
lr = 0.005
sgd = SGD(lr=lr, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#model.summary()
model.load_weights('8Train2TestStandartNetwork2.h5', by_name=True)

# read video
video = 'Explosion026_x264.mp4'
cap = cv2.VideoCapture(video)
file = open('deneme.txt','w')
clip = []
dizi = []
c_dizi = []
counter = 0
plt.figure(figsize=(6, 3))
# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
while True:
    ret, frame = cap.read()
    if ret:
        tmp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        clip.append(cv2.resize(tmp, (171, 128)))
        if len(clip) == 20:
            inputs = np.array(clip).astype(np.float32)
            inputs = np.expand_dims(inputs, axis=0)
            inputs[..., 0] -= 99.9
            inputs[..., 1] -= 92.1
            inputs[..., 2] -= 82.6
            inputs[..., 0] /= 65.8
            inputs[..., 1] /= 62.3
            inputs[..., 2] /= 60.3
            inputs = inputs[:,:,8:120,30:142,:]
            inputs = np.transpose(inputs, (0, 2, 3, 1, 4))
            pred = model.predict(inputs)
            # label = np.argmax(pred[0])
            index = np.argsort(pred[0, :])

            # print(class_names[index[3]])

            cv2.putText(frame, class_names[index[5]].strip(), (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 1)
            cv2.putText(frame, "prob: %.4f" % pred[0, index[5]], (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 1)
            temp = str(class_names[index[5]]).split(' ')
            #print(int(temp[0]))
            dizi.append(int(temp[0]))
            print(temp[0])
            file.write(temp[0] + "\n")
            #dizi[counter] =  class_names[index[5]]
            counter = counter+1
            c_dizi.append(counter)
            cv2.putText(frame, class_names[index[4]].strip(), (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 1)
            cv2.putText(frame, "prob: %.4f" % pred[0, index[4]], (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 1)
            cv2.putText(frame, class_names[index[3]].strip(), (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 1)
            cv2.putText(frame, "prob: %.4f" % pred[0, index[3]], (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 1)

            cv2.putText(frame, class_names[index[2]].strip(), (150, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 1)
            cv2.putText(frame, "prob: %.4f" % pred[0, index[2]], (150, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 1)

            cv2.putText(frame, class_names[index[1]].strip(), (150, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 1)
            cv2.putText(frame, "prob: %.4f" % pred[0, index[1]], (150, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 1)
            cv2.putText(frame, class_names[index[0]].strip(), (150, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 1)
            cv2.putText(frame, "prob: %.4f" % pred[0, index[0]], (150, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 1)

            clip.pop(0)
        cv2.imshow('result', frame)
        cv2.waitKey(1)
    else:
        break
cap.release()
cv2.destroyAllWindows()
file.close()
plt.plot(c_dizi, dizi)
plt.xlabel('Time')
plt.ylabel('Class')
plt.savefig('my_figure.png')