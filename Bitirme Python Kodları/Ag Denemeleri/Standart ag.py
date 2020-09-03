#standart ağ
from keras.layers import Dense,Dropout,Conv3D,Input,MaxPool3D,Flatten,Activation,ZeroPadding3D,AveragePooling3D
from keras.regularizers import l2
from keras.models import Model
import random
import numpy as np
import cv2
import os
from keras.optimizers import SGD, Adam, Adagrad
from keras.utils import np_utils
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from keras.layers import LeakyReLU
from keras.layers import add
import keras.backend as K
from keras.callbacks import Callback, ModelCheckpoint
import yaml
import h5py
import numpy as np
import keras.backend as K
from keras.callbacks import Callback, ModelCheckpoint
import yaml
import h5py
import numpy as np

class Step(Callback):

    def __init__(self, steps, learning_rates, verbose=0):
        self.steps = steps
        self.lr = learning_rates
        self.verbose = verbose

    def change_lr(self, new_lr):
        old_lr = K.get_value(self.model.optimizer.lr)
        K.set_value(self.model.optimizer.lr, new_lr)
        if self.verbose == 1:
            print('Learning rate is %g' %new_lr)

    def on_epoch_begin(self, epoch, logs={}):
        for i, step in enumerate(self.steps):
            if epoch < step:
                self.change_lr(self.lr[i])
                return
        self.change_lr(self.lr[i+1])

    def get_config(self):
        config = {'class': type(self).__name__,
                  'steps': self.steps,
                  'learning_rates': self.lr,
                  'verbose': self.verbose}
        return config

    @classmethod
    def from_config(cls, config):
        offset = config.get('epoch_offset', 0)
        steps = [step - offset for step in config['steps']]
        return cls(steps, config['learning_rates'],
                   verbose=config.get('verbose', 0))

def plot_history(history, result_dir):
    plt.plot(history.history['accuracy'], marker='.')
    plt.plot(history.history['val_accuracy'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_accuracy'], loc='lower right')
    plt.savefig(os.path.join(result_dir, 'model_accuracy.png'))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'model_loss.png'))
    plt.close()


def save_history(history, result_dir):
    loss = history.history['loss']
    acc = history.history['accuracy']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_accuracy']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, 'result.txt'), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))
        fp.close()

def c3d_model():
    input_shape = (112, 112, 20, 3)
    weight_decay = 0.005
    nb_classes = 101

    inputs = Input(input_shape)
    x = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu', kernel_regularizer=l2(weight_decay))(inputs)
    x = MaxPool3D((2, 2, 1), strides=(2, 2, 1), padding='same')(x)

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
    x = Dense(4, kernel_initializer='glorot_normal', kernel_regularizer=l2(0.001), activation='softmax')(x)

    model = Model(inputs, x)
    return model
def generator_train_batch(train_txt, batch_size, img_path):
    ff = open(train_txt, 'r')
    lines = ff.readlines()
    num = len(lines)
    while True:
        new_line = []
        index = [n for n in range(num)]
        random.shuffle(index)
        for m in range(num):
            new_line.append(lines[index[m]])
        for i in range(int(num / batch_size)):
            a = i * batch_size
            b = (i + 1) * batch_size
            x_train, x_labels = process_batch(new_line[a:b], img_path)

            x = preprocess(x_train)
            x = np.transpose(x, (0, 2, 3, 1, 4))
            y = np_utils.to_categorical(np.array(x_labels), 4)
            yield x, y

def generator_val_batch(val_txt, batch_size, img_path):
    f = open(val_txt, 'r')
    lines = f.readlines()
    num = len(lines)
    while True:
        new_line = []
        index = [n for n in range(num)]
        random.shuffle(index)
        for m in range(num):
            new_line.append(lines[index[m]])
        for i in range(int(num / batch_size)):
            a = i * batch_size
            b = (i + 1) * batch_size
            y_test, y_labels = process_batch(new_line[a:b], img_path)

            x = preprocess(y_test)
            x = np.transpose(x, (0, 2, 3, 1, 4))
            y = np_utils.to_categorical(np.array(y_labels), 4)
            yield x, y


def process_batch(lines, img_path):
    num = len(lines)
    batch = np.zeros((num, 20, 112, 112, 3), dtype='float32')
    labels = np.zeros(num, dtype='int')
    for i in range(num):  # num -- 16
        path = lines[i].split(' ')[0]
        label = lines[i].split(' ')[-1]
        label = label.strip('\n')
        label = int(label)
        frameNumber = int(lines[i].split(' ')[1])
        for j in range(20):
            try:
                image = cv2.imread(img_path  + 'frames/' + path + '/frame' + str((frameNumber)+j) + '.jpg')
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (171, 128))
                batch[i][j][:][:][:] = image[8:120, 30:142, :]
            except Exception as e:
                print("hata\n")
                print(img_path + 'Explosion/' + path + '/frame' + str((frameNumber) + j) + '.jpg')
        labels[i] = label
    return batch, labels
def preprocess(inputs):
    inputs[..., 0] -= 99.9
    inputs[..., 1] -= 92.1
    inputs[..., 2] -= 82.6
    inputs[..., 0] /= 65.8
    inputs[..., 1] /= 62.3
    inputs[..., 2] /= 60.3
    # inputs /=255.
    # inputs -= 0.5
    # inputs *=2.
    return inputs

def onetenth_4_8_12(lr):
    steps = [4, 8,12]
    lrs = [lr, lr/10, lr/100,lr/1000]
    return Step(steps, lrs)

img_path = 'C:/Users/onure/PycharmProjects/ImageClassification/'

train_file = 'C:/Users/onure/PycharmProjects/ImageClassification/4-1ButunVideolar20-15frameTrain.txt'
test_file = 'C:/Users/onure/PycharmProjects/ImageClassification/4-1ButunVideolar20-15frameTest.txt'

f1 = open(train_file, 'r')
f2 = open(test_file, 'r')
lines = f1.readlines()
f1.close()
train_samples = len(lines)
lines = f2.readlines()
f2.close()
val_samples = len(lines)

batch_size = 2
epochs = 50
learning_rate = 0.001

model = c3d_model()

sgd = SGD(lr=learning_rate, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss="categorical_crossentropy",metrics=['accuracy'])
model.summary()

history = model.fit_generator(generator_train_batch(train_file, batch_size, img_path),
                        steps_per_epoch=train_samples // batch_size,
                        epochs=epochs,
                        callbacks=[onetenth_4_8_12(learning_rate)],
                        validation_data=generator_val_batch(test_file,
                                                        batch_size, img_path),
                        validation_steps=val_samples // batch_size,
                        verbose=1)


model.save_weights('C:/Users/onure/PycharmProjects/ImageClassification/4-1Videolar-20-15framellbos.h5')
plot_history(history, 'C:/Users/onure/PycharmProjects/ImageClassification/')
save_history(history, 'C:/Users/onure/PycharmProjects/ImageClassification/')


