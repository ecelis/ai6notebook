# Deep Learning 2018 - Lesson 3

* [Lecture](https://youtu.be/9C06ZPF8Uuc)
* [Wiki](http://forums.fast.ai/t/wiki-lesson-3/9401)


## Links

* [Decoding the ResNet architecture](http://teleported.in/posts/decoding-resnet-architecture/)
* [Hiromi Suenaga's lesson 2 notes](https://medium.com/@hiromi_suenaga/deep-learning-2-part-1-lesson-3-74b0ef79e56)


## Notes

* **Activation** a number that is calculated by applying some kind of
  linear function to numbers on the input.

* **Activation Function** is a function that takes an input number and
  spits an activation.

* **Softmax** is used in last layer, outputs values between 0 and 1. And all
  outputs sum up to 1.
  - softamx wants to pick a thing, not good for multi label images.

* **Sigmoid** works better for multi-label classification.

* **Rectifier Linear Unit (ReLU)** linear function used mostrly on
  output layers.

* Activation functions have personalities.

* `ConvLearner.pretrained(nn_model, data)` builds a learner that
  contains a pre-trained model.  The last layer of the model needs to be
  replaced with the layer of the right dimensions. The pretained model was
  trained for 1000 classes therfore the final layer predicts a vector of
  1000 probabilities.

* Parameters are learned by fitting a model to the data. Hyperparameters
  are another kind of parameter, that cannot be directly learned from the
  regular training process. These parameters express “higher-level”
  properties of the model such as its complexity or how fast it should
  learn. Two examples of hyperparameters are the learning rate and the
  number of epochs.

* During iterative training of a neural network, a batch or mini-batch is
  a subset of training samples used in one iteration of Stochastic
  Gradient Descent (SGD). An epoch is a single pass through the entire
  training set which consists of multiple iterations of SGD.

* We can now fit the model; that is, use gradient descent to find the best
  parameters for the fully connected layer we added, that can separate cat
  pictures from dog pictures.
  - `precompute=True` is a shortcut that catches the intermediate steps
    so we don't need to recalculate later. If enabled data augmentations
  does nothing.

* If using a data set similar to ImageNet on resnext50 or resnext101,
  that is images 340px wide, side photos, standars objects, etc... After
  `learner.unfreeze()` you should call `learner.bn_freeze(True)` it causes
  the _batch normalization moving averages_ to not be updated.

* It is important to understand how to use libraries other than Fast.ai.
  [Keras](http://keras.io) is a good example to look at because just
  like Fast.ai sits on top of PyTorch, it sits on top of varieties of
  libraries such as TensorFlow, MXNet, CNTK, etc.
  - Compared with [fastai](https://github.com/fastai/fastai), Keras
    requires much more code and many more parameters to be set.
  - Rather than creating a single data object, in Keras you define
    DataGenerator and specify what kind of data augmentation we want it
    to do and also what kind of normalization to do. In other words, in
    Fast.ai, we can just say “whatever ResNet50 requires, just dot hat for
    me please” but in Keras, you need to know what is expected. There is no
    standard set of augmentations.
  - You have to then create a validation data generator in which you are
    responsible to create a generator that does not have data
    augmentation. And you also have to tell it not to shuffle the dataset
    for validation because otherwise you cannot keep track of how well you
    are doing.

* When submitting results to competitions set `log_preds, y =
  learn.TTA(is_test=True)`, it will give predictions on the test set
  rather than in the validation set.

* By default, PyTorch models will give you back the log of the
  predictions, so you need to do `np.exp(log_preds)` to get the
  probability.

* To predict on a single image:

Shortest wasy to get a prediction:


        trn_tfms, val_tfms = tfms_from_model(arch, sz)
        im = val_tfms(Image.open(PATH+fn)
        preds = learn.predict_array(im[None])
        np.argmax(preds)


* Image must be transformed. tfms_from_model returns training
  transforms and validation transforms. In this case, we will use
  validation transform.

* Everything that gets passed to or returned from a model is generally
  assumed to be in a mini-batch. Here we only have one image, but we have
  to turn that into a mini-batch of a single image. In other words, we
  need to create a tensor that is not just [rows, columns, channels] , but
  [number of images, rows, columns, channels].

* im[None] : Numpy trick to add additional unit axis to the start.


* pytorch concepts
  - `ds` = data set
  - `dl` = data loader, works on minibatches

* Convolution is something where we have a little matrix (nearly always
  3x3 in deep learning) and multiply every element of that matrix by
  every element of 3x3 section of an image and add them all together to
  get the result of that convolution at one point.
  - [Visual understanding of Deep Learning](https://youtu.be/9C06ZPF8Uuc?t=49m51s)
  - [Convolution on Exel](https://docs.google.com/spreadsheets/d/1kkUTVzKE1Xmi24MadQDszBIHXagH8_YYMUSC4Pvwqec/edit?usp=sharing)

* **Filter/Kernel** is a 3x3 slice of a 3D tensor used for convolution.

* **Tensor** is a multi-dimesional array or matrix.

* **Hidden layer** is an internal layer, neither input or output layer.

* **Maxpooling** a (2,2) maxpooing ???

* **Fully connected layer** Gives a weight to each and every single
  activation and calculate the sum product. Weight matrix is as big as
  the entire input.

* Data
  - Unstructured — Audio, images, natural language text where all of
    the things inside an object are all the same kind of
    things — pixels, amplitude of waveform, or words.
  - Structured — Profit and loss statement, information about a Facebook
    user where each column is structurally quite different. “Structured”
    refers to columnar data as you might find in a database or a spreadsheet
    where different columns represent different kinds of things, and each
    row represents an observation.

* `fastai.structured` — not PyTorch specific and also used in machine
  learning course doing random forests with no PyTorch at all. It can used
  on its own without any of the other parts of Fast.ai library.

* `fastai.column_data` — allows us to do Fast.ai and PyTorch stuff with
  columnar structured data.
