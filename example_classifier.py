from keras.models import Model
from keras.layers import Input, Dense
from keras.regularizers import l1
from keras.utils import to_categorical
from keras.callbacks import Callback
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from BSFilter import BSFilter
import numpy as np
import matplotlib.pyplot as plt

#####################################
# callback for visualization purposes
#####################################


class BSFLogger(Callback):
    def on_train_begin(self, logs={}):
        self.weights = []

    def on_epoch_end(self, batch, logs={}):
        layer = self.model.get_layer("bs_filter_1")
        self.weights.append(layer.get_weights()[0])

#############
# load data
#############


X, Y = load_wine(return_X_y=True)  # could be changed to any from sklearn.datasets
scaler = StandardScaler()
X = scaler.fit_transform(X)
Y = to_categorical(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.30,
                                                    random_state=31525)  # for repeatability
######################
# define and fit model
######################

dim = X_train.shape[1]
num_of_classes = Y_train.shape[1]
penalty_coef = 0.03

inp = Input(shape=(dim,))
l = BSFilter(regularizer=l1(penalty_coef), initializer=0.99)(inp)
l = Dense(dim, activation="relu")(l)
l = Dense(dim, activation="relu")(l)

output = Dense(num_of_classes, activation='softmax')(l)
model = Model(inp, output)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
bsflogger = BSFLogger()
history = model.fit(x=X_train, y=Y_train, batch_size=32, epochs=1500,
                    validation_data=[X_test, Y_test], callbacks=[bsflogger])

#################################
# print importances and visualize
#################################

bsf_weights = model.layers[1].get_weights()[0]
print("\n\nFeature # \t Importance")
for idx, i in enumerate(bsf_weights):
    print("{}\t{:.3}".format(idx, i))

plt.subplot(2, 1, 1)
plt.plot(history.history['loss'], "k-", label="Loss")
plt.plot(history.history['val_loss'], "g-", label="Validation loss")
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(history.history['acc'], "k-", label="Accuracy")
plt.plot(history.history['val_acc'], "g-", label="Validation accuracy")
plt.plot(np.sum(bsflogger.weights, axis=1) / dim, "b--", label="Mean feature involvement")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.imshow(bsflogger.weights, aspect=0.02)
plt.xlabel("Feature")
plt.ylabel("Training epoch")
plt.xticks(np.arange(dim, step=2))
cbar = plt.colorbar()
cbar.set_label("Feature importance")
plt.tight_layout()
plt.show()
