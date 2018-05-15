# Deep Learning 2018 - Lesson 2

    lr = np.array([1e-4,1e-3,1e-2])

# Questions

val_idxs what triggers exactly?


# Links

* [**ERROR:** TypeError: torch.max received an invalid combination of arguments - got (numpy.ndarray, dim=int)](http://forums.fast.ai/t/lesson1-ipynb-error-typeerror-torch-max-received-an-invalid-combination-of-arguments-got-numpy-ndarray-dim-int/10707/3?u=ecelis)


# Notes

* models, h5 saved files
* sample, TODO
* test1, TODO
* tmp, precomputed TODO
* train, TODO
* valid, cats and dogs


`learn.fit`, gets an instance of LayerOptimizer and delegates itself to
`self.fit_gen()`. You can specify a list of learning rates which will be
applied to differenet segments of an architecture. This seems mostly
relevant to ImageNet-trained models, where we want to alter the layers
closest to the images by much smaller amounts.

- lrs (float, or float list) learning rate for the model.
- n_cycle int number of cycles (iterations) to fit the model
- wds float or list float weight decay parameter(s)

The number of epochs between resetting the learning rate is set by
`cycle_len`, and the number of times this happens is refered to as the
number of cycles, and is what we're actually passing as the 2nd
parameter to `fit()`

The learning rate determines how quickly or how slowly you want to
update the weights (or parameters). Learning rate is one of the most
difficult parameters to set, because it significantly affect model
performance.

The method `learn.lr_find()` helps you find an optimal learning rate. It
uses the technique developed in the 2015 paper [Cyclical Learning Rates
for Training Neural Networks](http://arxiv.org/abs/1506.01186)

We first create a new learner, since we want to know how to set the
learning rate for a new (untrained) model.

Our learn object contains an attribute `sched` that contains our learning
rate scheduler, and has some convenient plotting functionality.


    learn.sched.plot_lr() learn.sched.plot()


If you try training for more epochs, you'll notice that we start to
overfit, which means that our model is learning to recognize the
specific images in the training set, rather than generalizing such that
we also get good results on the validation set. One way to fix this is


    learn.precompute = False


By default when we create a learner, it sets all but the last layer to
frozen. That means that it's still only updating the weights in the last
layer when we call fit.

to effectively create more data, through data augmentation. This refers
to randomly changing the images in ways that shouldn't impact their
interpretation, such as horizontal flipping, zooming, and rotating.

We can do this by passing `aug_tfms` (augmentation transforms) to
`tfms_from_model`, with a list of functions to apply that randomly change
the image however we wish. For photos that are largely taken from the
side (e.g. most photos of dogs and cats, as opposed to photos taken from
the top down, such as satellite imagery) we can use the pre-defined list
of functions transforms_side_on. We can also specify random zooming of
images up to specified scale by adding the `max_zoom` parameter.

Another trick is adding the cycle_mult parameter.

A common way to analyze the result of a classification model is to use a
[confusion matrix](http://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/).
Scikit-learn has a convenient function we can use for this purpose.

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

ConvLearner.pretrained builds learner that contains a pre-trained model.
The last layer of the model needs to be replaced with the layer of the
right dimensions. The pretained model was trained for 1000 classes
therfore the final layer predicts a vector of 1000 probabilities.

Parameters are learned by fitting a model to the data. Hyperparameters
are another kind of parameter, that cannot be directly learned from the
regular training process. These parameters express “higher-level”
properties of the model such as its complexity or how fast it should
learn. Two examples of hyperparameters are the learning rate and the
number of epochs.

During iterative training of a neural network, a batch or mini-batch is
a subset of training samples used in one iteration of Stochastic
Gradient Descent (SGD). An epoch is a single pass through the entire
training set which consists of multiple iterations of SGD.

We can now fit the model; that is, use gradient descent to find the best
parameters for the fully connected layer we added, that can separate cat
pictures from dog pictures.

