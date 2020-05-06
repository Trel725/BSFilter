# BSFilter
Implementation of Binary Stochastic Filtering layer in Keras with Tensorflow backend.  
See https://arxiv.org/abs/1902.04510 for details.

# BSFilter2
An updated version rewritten in TensorFlow 2 (and tf.keras). Implements few improvements, such as 
stabilized training and suport for weight sharing. Recommended to use.
```
BSFilter(regularizer=None, initializer=0.5, share_axis=None, threshold=0.1)

 regularizer: regularizer to use, l1 is recommended
 initializer: constant value to initialize the weights, not the class instance
 share_axis: axis, along which filtering coefficients will be shared.
             it is mainly useful e.g. to force network select same features for 
             every channel (for that set share_axis to -1). Batch axis is not counted.
  threshold: used at prediction phase, features with weight lower than 
            threshold are determinstically dropped, with higher values
            are passed
```  
## Brief summary
This layer randomly passes or drops features with probabilities, equal to its weights. If layer is penalized
with L1 penalty, network efficiently converges to some lower number of features, while penalization
coefficient controls balance between minimization of features involving and goodness of fit. Good results
were achieved with penalization coefficient lying in the range 0.001-0.01, but tuning for specific case
might be needed. In general, if accuracy drops significantly after introducing BSFilter layer in comparison
to vanilla classifier, the coefficient needs to be lowered. On the other hand, if after convergence
all weights of BSFilter are much higher than 0 than either coefficient needs to be increased or it is impossible
to find feature subset that would not lead to the accuracy decrease, which was actualy observed on some datasets.

Below are waterfall plot showing evolution of BSFilter layer weights during network training for Wine and Musk2 datasets.
(https://archive.ics.uci.edu/ml/index.php)  

Wine                       |  Musk2
:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/Trel725/BSFilter/master/etc/Evolution_wine.png)  |  ![](https://raw.githubusercontent.com/Trel725/BSFilter/master/etc/Evolution_musk.png)

### Neuron pruning

Additionally, BSFilter layer could be inserted between dense layers to filter the connections between them and thus 
to perform neuron pruning. The function *prune*, available in etc/prune.py is able to remove BSFilter layers
and unimportant neurons from this model after training producing optimized model.
This function relies on keras-surgeon  
https://github.com/BenWhetton/keras-surgeon

## Usage

```
BSFilter(regularizer=None, initializer=0.5)

regularizer: usually keras.regularizers.l1, but any could be used.  
initializer: initial value of weights, 0.5 by default. Must be in range [0, 1] 
```
## Examples

Simple multilayer perceptron classifier with input feature filtering for selection. See example_classifier.py for details.
```python
from keras.models import Sequential
from keras.layers import *
from keras.regularizers import l1
from BSFilter import BSFilter
dim = 10
num_of_classes = 3
penalty_coef = 0.05

model = Sequential()
model.add(BSFilter(regularizer=l1(l=penalty_coef), input_shape=(dim,)))
model.add(Dense(dim, activation='relu'))
model.add(Dense(dim, activation='relu'))
model.add(Dense(num_of_classes, activation='softmax'))
```

Simple convolutional classifier, written with Functional API.

```python
from keras.models import Model
from keras.layers import *
from keras.regularizers import l1
from BSFilter import BSFilter
dim = 10
num_of_classes = 3
penalty_coef = 0.05

inp = Input(shape=(dim,))
l = BSFilter(regularizer=l1(penalty_coef), initializer=0.99)(inp)
l = Reshape((dim, 1))(l)
l = Conv1D(filters=8, kernel_size=8)(l)
l = LeakyReLU()(l)
l = Flatten()(l)
l = Dense(dim, activation="relu")(l)

output = Dense(num_of_classes, activation='softmax')(l)
model = Model(inp, output)
```


## Citing

If this layer was helpful, please cite following paper:

```
@article{trelin2019binary,
         title = {Binary Stochastic Filtering: a Solution for Supervised Feature Selection and Neural Network Shape Optimization},
         author = {Trelin, Andrii and Prochazka, Ales},
         journal = {arXiv preprint arXiv: 1902.04510},
         year = {2019}
         }
```
