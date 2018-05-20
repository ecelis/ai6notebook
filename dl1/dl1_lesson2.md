# Deep Learning 2018 - Lesson 2

* [Lecture](https://youtu.be/JNxcznsrRb8)
* [Wiki](http://forums.fast.ai/t/wiki-lesson-2/9399)


## Questions


## Links

* [**ERROR:** TypeError: torch.max received an invalid combination of arguments - got (numpy.ndarray, dim=int)](http://forums.fast.ai/t/lesson1-ipynb-error-typeerror-torch-max-received-an-invalid-combination-of-arguments-got-numpy-ndarray-dim-int/10707/3?u=ecelis)
* [Hiromi Suenaga's lesson 2 notes](https://medium.com/@hiromi_suenaga/deep-learning-2-part-1-lesson-2-eeae2edd2be4)

## Notes

* Convolutional neural network have these things called “activations.”
  An activation is a number that says “this feature is in this place
  with this level of confidence (probability)”

* `models` directory, h5 saved files

* The learning rate determines how quickly or how slowly you want to
  update the weights (or parameters). Learning rate is one of the most
  difficult parameters to set, because it significantly affect model
  performance.
  - A value too small will take a long time to get to the best value
  - If value too big it maight oscillate away from best value
  - Learning rate finder `learn.lr_find` will increase the learning rate
    after each mini-batch.
  - The method `learn.lr_find()` helps you find an optimal learning rate. It
    uses the technique developed in the 2015 paper [Cyclical Learning Rates
    for Training Neural Networks](http://arxiv.org/abs/1506.01186)

* We first create a new learner, since we want to know how to set the
  learning rate for a new (untrained) model.

* Our learn object contains an attribute `sched` that contains our learning
  rate scheduler, and has some convenient plotting functionality.
  `learn.sched.plot_lr()` and `learn.sched.plot()`

* By default when we create a learner, it sets all but the last layer to
  frozen. That means that it's still only updating the weights in the last
  layer when we call fit.

* Mini-batch is a set of few images we look at each time so that we are
  using the parallel processing power of the GPU effectively (generally
  64 or 128 images at a time)

* Overfitting — the model is starting to see the specific details of the
  images in the training set rather than learning something general that
  can be transferred across to the validation set

* _Data augmentation_, every epoch, we will randomly change the image a
  little bit. In other words, the model is going to see slightly
  different version of the image each epoch.

* If you try training for more epochs, you'll notice that we start to
  overfit, which means that our model is learning to recognize the
  specific images in the training set, rather than generalizing such that
  we also get good results on the validation set. One way to fix this is
  setting `learn.precompute = False`, it must be done in order to use data
  augmentation.
* `learn.fit`, gets an instance of LayerOptimizer and delegates itself to
  `self.fit_gen()`. You can specify a list of learning rates which will be
  applied to differenet segments of an architecture. This seems mostly
  relevant to ImageNet-trained models, where we want to alter the layers
  closest to the images by much smaller amounts.
  - `learn.fit(lrs, n_cycle, wds [, cycle_len])` outputs _epoch number, -training loss, validation loss and
    accuracy_.
  - lrs (float, or float list) learning rate for the model.
  - n_cycle int number of cycles (iterations) to fit the model
  - wds float or list float weight decay parameter(s)
  - The number of epochs between resetting the learning rate is set by
    `cycle_len`, enables stochastic gradient descent with restarts
    (SGDR), and the number of times this happens is refered to as the
    number of cycles, and is what we're actually passing as the 2nd
    parameter to `fit()`.
    `cycle_len=2`, it will do 3 cycles where each cycle is 2 epochs (i.e.
    6 epochs).
  - `cycle_mult=2` this multiplies the length of the cycle after each
    cycle (1 epoch + 2 epochs + 4 epochs = 7 epochs).
  - there is a parameter called `cycle_save_name` which you can add as
    well as `cycle_len`, which will save a set of weights at the end of
    every learning rate cycle and then you can ensemble them.

* Data uagmentation is done by passing `aug_tfms` (augmentation
transforms) to `tfms_from_model`, with a list of functions to apply that
randomly change the image however we wish. For photos that are largely
taken from the side (e.g. most photos of dogs and cats, as opposed to
photos taken from the top down, such as satellite imagery) we can use
the pre-defined list of functions transforms_side_on. We can also
specify random zooming of images up to specified scale by adding the
`max_zoom` parameter.

* Another trick is adding the `cycle_mult` parameter.

* A common way to analyze the result of a classification model is to use a
  [confusion matrix](http://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/).
  Scikit-learn has a convenient function we can use for this purpose.

* “frozen” layer is a layer which is not being trained/updated.
  `learn.unfreeze()` unfreezes all the layers.

* `learn.TTA()` “Test Time Augmentation”. What this means is that we are
  going to take 4 data augmentations at random as well as the
  un-augmented original (center-cropped). We will then calculate
  predictions for all these images, take the average, and make that our
  final prediction. Note that this is only for validation set and/or test
  set.

* In order to train the model, data needs to be organized in a certain
  way (example below uses dogs vs cats data set).

Under `train` and `valid` directories images must be classified in
sub-directories named after each class.

`learn.save()` output model goes to `models`


      /home/paperspace/data/dogscats
      ├── models
      ├── sample
      │   ├── train
      │   │   ├── cats
      │   │   └── dogs
      │   └── valid
      │       ├── cats
      │       └── dogs
      ├── test1
      ├── tmp
      │   ├── x_act_resnet34_0_224.bc
      │   │   ├── data
      │   │   └── meta
      │   ├── x_act_test_resnet34_0_224.bc
      │   │   ├── data
      │   │   └── meta
      │   └── x_act_val_resnet34_0_224.bc
      │       ├── data
      │       └── meta
      ├── train
      │   ├── cats
      │   └── dogs
      └── valid
          ├── cats
          └── dogs


## Review: easy steps to train a world-class image classifier

1. Enable data augmentation, and `precompute=True`
1. Use `lr_find()` to find highest learning rate where loss is still
clearly improving
1. Train last layer from precomputed activations for 1-2 epochs
1. Train last layer with data augmentation (i.e. precompute=False) for 2-3 epochs with cycle_len=1
1. Unfreeze all layers
1. Set earlier layers to 3x-10x lower learning rate than next higher
layer
1. Use `lr_find()` again
1. Train full network with `cycle_mult=2` until over-fitting

`ConvLearner.pretrained` builds a learner that contains a pre-trained model.
The last layer of the model needs to be replaced with the layer of the
right dimensions. The pretained model was trained for 1000 classes
therfore the final layer predicts a vector of 1000 probabilities.

**Parameters are learned by fitting a model to the data**.
**Hyperparameters** are another kind of parameter, that **cannot be
directly learned from the regular training process**. These parameters
express “higher-level” properties of the model such as its complexity or
how fast it should learn. Two examples of hyperparameters are the
learning rate and the number of epochs.

During iterative training of a neural network, a **batch or mini-batch
is a subset of training samples used in one iteration of Stochastic
Gradient Descent (SGD)**. An **epoch is a single pass through the entire
training set which consists of multiple iterations of SGD**.

We can now fit the model; that is, use gradient descent to find the best
parameters for the fully connected layer we added, that can separate cat
pictures from dog pictures.
