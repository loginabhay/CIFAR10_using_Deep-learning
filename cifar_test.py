from matplotlib import pyplot
from scipy.misc import toimage
from keras.datasets import cifar10
import numpy as np
from keras.models import model_from_json
from sklearn.metrics import classification_report

def show_imgs(X):
    pyplot.figure(1)
    k = 0
    for i in range(0,4):
        for j in range(0,4):
            pyplot.subplot2grid((4,4),(i,j))
            pyplot.axis('off')
            pyplot.imshow(toimage(X[k]))
            k = k+1
    # show the plot
    pyplot.show()
model = load_('model.h5')
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
predicted_classes = model.predict(x_test)

# mean-std normalization
mean = np.mean(x_train ,axis=(0 ,1 ,2 ,3))
std = np.std(x_train ,axis=(0 ,1 ,2 ,3))
x_train = (x_train -mean ) /(std +1e-7)
x_test = (x_test -mean ) /(std +1e-7)

show_imgs(x_test[:16])

# Load trained CNN model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('model.h5')

labels =  ['airplane' ,'automobile' ,'bird' ,'cat' ,'deer' ,'dog' ,'frog' ,'horse' ,'ship' ,'truck']

indices = np.argmax(model.predict(x_test[:16]) ,1)
print ([labels[x] for x in indices])
num_classes = 10
target_names = ["Classe {}".format(i) for i in range(num_classes)]
print(classification_report(y_test, predicted_classes, target_names=target_names))