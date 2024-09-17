import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset

# Creating tensors  in PyTorch
np.set_printoptions(precision=3)
a = [1, 2, 3]
b = np.array([4, 5, 6], dtype=np.int32)
t_a = torch.tensor(a)
t_b = torch.from_numpy(b)
print(t_a)
print(t_b)

t_ones = torch.ones(2, 3)
t_ones.shape
print(t_ones)

rand_tensor = torch.rand(2, 3)
print(rand_tensor)

# ### Manipulating the data type and shape of a tensor

# Change data type of tensor to desired type: torch.to()
t_a_new = t_a.to(torch.int64)
print(t_a_new.dtype)

# Transposing a tensor: torch.transpose()
t = torch.rand(3, 5)
t_tr = torch.transpose(t, 0, 1)
print(t.shape, ' --> ', t_tr.shape)

# Reshaping a tensor (i.e.  from 1D vector to 2D array): torch.reshape()
t = torch.zeros(30)
t_reshape = t.reshape(5, 6)
print(t_reshape.shape)

# Removing dimensions (i.e. dimensions with size 1, not needed)
t = torch.zeros(1, 2, 1, 4, 1)
t_sqz = torch.squeeze(t, 2)
print(t.shape, ' --> ', t_sqz.shape)


# ### Applying mathematical operations to tensors

# instantiate two random tensors, one with uniform distribution in range [-1, 1)
# and the other with a standard normal distribution:
torch.manual_seed(1)
t1 = 2 * torch.rand(5, 2) - 1 # torch.rand returns tensor filled with random  numbers from uniform distribution in range [0, 1)
t2 = torch.normal(mean=0, std=1, size=(5, 2)) # note t1 and t2 have same shape

# Compute element-wise product of t1 and t2
t3 = torch.multiply(t1, t2)
print(t3)

# Compute mean, sum and standard deviation along a certain axis (or axes)
# mean(), torch.sum(), and torch.std(). Mean each of column in t1 can be computed:
t4 = torch.mean(t1, axis=0)
print(t4)

# matrix-matrix product between t1 and t2 (that is t_1 × (t_2)^T where the superscript T is for transpose)
# computed bby using torch.matmul()
t5 = torch.matmul(t1, torch.transpose(t2, 0, 1))
print(t5)

# compute (t_1)^T × t_2 performed by transposing t1 resulting in 2x2 array
t6 = torch.matmul(torch.transpose(t1, 0, 1), t2)
print(t6)

# Compute L^p norm of a tensor with torch.linalg.norm(). I.e. L^2 norm of t1:
norm_t1 = torch.linalg.norm(t1, ord=2, dim=1)
print(norm_t1)

# Verify code snippet computing L^2 norm of t1 correctly, compare with NumPy function:
print(np.sqrt(np.sum(np.square(t1.numpy()), axis=1)))


# ### Split, stack, and concatenate tensors

# Divide input tensor into a list of equally sized tensors: torch.chunk()
# use chunks argument to determine desired number of splits as an integer to split a tensor along the desired dimension
# specified dimension must be divisble by the desired number of splits. Alternatively we can provide
# the desired sizes in a list using the torch.split() function:

# provide  number of splits:
torch.manual_seed(1)
t = torch.rand(6)
print(t)

t_splits = torch.chunk(t, 3)
[item.numpy() for item in t_splits]
# in this example a tensor of size 6 was divided into a list of three tensors each with size 2
# if the tensor size is not divisible by the chunks value, the last cunk will be smaller

# providing the sizes of different splits: alternatively we specify sizes of the output tensor directly
# instead of number of splits
torch.manual_seed(1)
t = torch.rand(5)
print(t)
t_splits = torch.split(t, split_size_or_sections=[3, 2])
[item.numpy() for item in t_splits]
# here we split tensor of size 5 into tensors of sizes 3 and 2


# Concatenate / stack multiple tensors to create a single tensor
# Create 1D tensor, A, containing 1s with size 3, and a 1D tensor, B, containing 0s with size 2
# and concatenate them into a 1D tensor, C, of size 5:
A = torch.ones(3)
B = torch.zeros(2)
C = torch.cat([A, B], axis=0)
print(C)

# Create 1D tensors A and B, both size 3, and stack them together to form a 2D tensor S:
A = torch.ones(3)
B = torch.zeros(3)
S = torch.stack([A, B], axis=1)
print(S)
print(1)


# ## Building input pipelines in PyTorch

# ### Creating a PyTorch DataLoader from existing tensors
t = torch.arange(6, dtype=torch.float32)
data_loader = DataLoader(t)

for item in data_loader:
    print(item)

data_loader = DataLoader(t, batch_size=3, drop_last=False)
for i, batch in enumerate(data_loader, 1):
    print(f'batch {i}:', batch)

# ### Combining two tensors into a joint dataset
# Often we may have data in two or more tensors and need to combine them, to retrieve elements in tuples
torch.manual_seed(1)
t_x = torch.rand([4, 3], dtype=torch.float32) # hold feature values, each size 3
t_y = torch.arange(4) # hold class labels

class JoinDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# Note a custom Dataset class must contain the following methods to be used by the data loader later:
# 1. __init__(): initial logic, such as reading existing arrays, loading a file, filtering data etc.
# 2. __getitem__(): returns corresponding sample to the given index

joint_dataset = JoinDataset(t_x, t_y)

for example in joint_dataset:
    print(' x: ', example[0], ' y: ', example[1])

# We can also simply utilize torch.utils.data.TensorDataset class, if the second dataset is a labele dataset
# in the form of tensors. So instead of using our self-defined Dataset class, JointDataset, we can create a joint dataset as

# TensorDataset directly
joint_dataset = TensorDataset(t_x, t_y)

for example in joint_dataset:
    print('  x: ', example[0],
          '  y: ', example[1])


# ### Shuffle, batch, and repeat
torch.manual_seed(1)
data_loader = DataLoader(dataset=joint_dataset, batch_size=2, shuffle=True)

for i, batch in enumerate(data_loader, 1):
    print(f'batch {i}:', 'x:', batch[0], '\n      y:', batch[1])

# Ideally when training a model for multiple epoch,
# we need to shuffle and iterate over the dataset by the desired number of epochs
for epoch in range(2):
    print(f'epoch {epoch+1}')
    for i, batch in enumerate(data_loader, 1):
        print(f'batch {i}:', 'x:', batch[0], '\n      y:', batch[1])


# ### Creating a dataset from files on your local storage disk
import pathlib
import matplotlib.pyplot as plt
import os
from PIL import Image
imgdir_path = pathlib.Path('cat_dog_images')
file_list = sorted([str(path) for path in imgdir_path.glob('*.jpg')])
print(file_list)

fig = plt.figure(figsize=(10, 5))
for i, file in enumerate(file_list):
    img = Image.open(file)
    print('Image shape:', np.array(img).shape)
    ax = fig.add_subplot(2, 3, i+1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(img)
    ax.set_title(os.path.basename(file), size=15)
plt.tight_layout()
plt.show()

# preprocess images to consistent size, so we don't have different sized aspect ratios
# Labels on images are provided within their filenames, instead we extract the labels from the list of filenames
# assigning label 1 to dogs and label 0 to cats.
labels = [1 if 'dog' in os.path.basename(file) else 0 for file in file_list]
print(labels)

# Now we have a list of filenames (path of each image) and list of their labels
# Now we will create joint dataset from two arrays

class ImageDataset(Dataset):
    def __init__(self, file_name, labels):
        self.file_list = file_list
        self.labels = labels

    def __getitem__(self, index):
        file = self.file_list[index]
        label = self.labels[index]
        return file, label

    def __len__(self):
        return len(self.labels)

image_dataset = ImageDataset(file_list, labels)
for file, label in image_dataset:
    print(file, label)

# Transform dataset: load image content from its file path, decode the raw content and resize to desired size, i.e., 80x120
# use torcvision.transform module to resize image and convert loaed pixels into tensors
import torchvision.transforms as transforms
img_height, img_width = 80, 120
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((img_height, img_width)),
])

# update ImageDataset with transform as we defined
class ImageDataset(Dataset):
    def __init__(self, file_list, labels, transform=None):
        self.file_list = file_list
        self.labels = labels
        self.transform = transform
    def __getitem__(self, index):
        img = Image.open(self.file_list[index])
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        return img, label
    def __len__(self):
        return len(self.labels)

image_dataset = ImageDataset(file_list, labels, transform)

fig = plt.figure(figsize=(10, 6))
for i, example in enumerate(image_dataset):
    ax = fig.add_subplot(2, 3, i+1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(example[0].numpy().transpose((1, 2, 0)))
    ax.set_title(f'{example[1]}', size=15)
plt.tight_layout()
plt.show()


# ### Fetching available datasets from the torchvision.datasets library

# **Fetching CelebA dataset**
#
# ---

# 1. Downloading the image files manually
#
# - You can try setting `download=True` below. If this results in a `BadZipfile` error, we recommend downloading the
# `img_align_celeba.zip` file manually from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html. In the Google Drive folder,
# you can find it under the `Img` folder as shown below:
import torchvision
from itertools import islice

image_path = './'
celeba_dataset = torchvision.datasets.CelebA(image_path, split='train', target_type='attr', download=False) # Already downloaded once, so download=False

assert isinstance(celeba_dataset, torch.utils.data.Dataset)

example = next(iter(celeba_dataset))
print(example)

# Take first 18 examples from CelebA and visualize with their 'Smiling' label whether they smile or not (1 or 0)
fig = plt.figure(figsize=(12, 8))
for i, (image, attributes) in islice(enumerate(celeba_dataset), 18):
    print(i)
    print((image, attributes))
    ax = fig.add_subplot(3, 6, i+1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(image)
    ax.set_title(f'{attributes[31]}', size=15) # CelebA has 40 attributes per image. The smiling label is located at index 31. So attributes[31] extracts 'Smiling' label for each image, and it's either 1 (person is smiling) or 0 (person is not smiling).
plt.show()

# Download the 'train' partition, convert the elements to tuples and visualize 10 examples:
mnist_dataset = torchvision.datasets.MNIST(image_path, 'train', download=False) # Already downloaded once, so download=False
assert isinstance(mnist_dataset, torch.utils.data.Dataset)
example = next(iter(mnist_dataset))
print(example)
fig = plt.figure(figsize=(15, 6))
for i, (image, label) in islice(enumerate(mnist_dataset), 10):
    ax = fig.add_subplot(2, 5, i+1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(image, cmap='gray_r')
    ax.set_title(f'{label}', size=15)
plt.show()

# ## Building a neural network model in PyTorch

# ### The PyTorch neural network module (torch.nn)

# ### Building a linear regression model
from torch.utils.data import TensorDataset

X_train = np.arange(10, dtype='float32').reshape((10, 1))
y_train = np.array([1.0, 1.3, 3.1, 2.0, 5.0, 6.3, 6.6,
                    7.4, 8.0, 9.0], dtype='float32')
plt.plot(X_train, y_train, 'o', markersize=10)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Standardize features (mean centering and dividing by standard deviation) and create PyTorch dataset for
# training set and corresponding DataLoader
X_train_norm = (X_train - np.mean(X_train)) / np.std(X_train)
X_train_norm = torch.from_numpy(X_train_norm)
y_train = torch.from_numpy(y_train)
train_ds = TensorDataset(X_train_norm, y_train)
batch_size = 1
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

# Now we can define our model for linear regression z = wx + b. For this we use torch.nn module
# which provides predefined layers for building complex NN models, but to start, we first will
# define how to model from scratch

# define model
torch.manual_seed(1)
weight = torch.randn(1)
weight.requires_grad_()
bias = torch.zeros(1, requires_grad=True)
def model(xb):
    return xb @ weight + bias

# Define loss function: here mean squared error (MSE)
def loss_fn(input, target):
    return (input-target).pow(2).mean()

# Use stochastic gradient descent to learn weight parameters of the model
# We will  implement this training via stochastic gradient decscent procedure ourselves.
# and next use the SGD method from the optimization package, torch.optim

# First we need to compute the gradients to implement SGD algorithm.
# rather than manually computing we will use PyTorch's torch.autograd.backward function

learning_rate = 0.001
num_epochs = 200
log_epochs = 10
for epoch in range(num_epochs):
    for x_batch, y_batch in train_dl:
        pred = model(x_batch)
        loss = loss_fn(pred, y_batch)
        loss.backward()

        with torch.no_grad():
            weight -= weight.grad * learning_rate
            bias -= bias.grad * learning_rate
            weight.grad.zero_()
            bias.grad.zero_()
    if epoch % log_epochs==0:
        print(f'Epoch {epoch}  Loss {loss.item():.4f}')

# plot trained model
# for test data we create NumPy array of values evenly spaced between 0 and 9
# since we trained our model with standardized features, we will also apply standardization to the test data
print('Final Parameters:', weight.item(), bias.item())

X_test = np.linspace(0, 9, num=100, dtype='float32').reshape(-1, 1)
X_test_norm = (X_test - np.mean(X_train)) / np.std(X_train)
X_test_norm = torch.from_numpy(X_test_norm)
y_pred = model(X_test_norm).detach().numpy()

fig = plt.figure(figsize=(13, 5))
ax = fig.add_subplot(1, 2, 1)
plt.plot(X_train_norm, y_train, 'o', markersize=10)
plt.plot(X_test_norm, y_pred, '--', lw=3)
plt.legend(['Training examples', 'Linear Reg.'], fontsize=15)
ax.set_xlabel('x', size=15)
ax.set_ylabel('y', size=15)
ax.tick_params(axis='both', which='major', labelsize=15)
plt.show()

# ### Model training via the torch.nn and torch.optim modules
import torch.nn as nn

# Create new MSE loss function and SGD optimizer with torch.optim and nn
loss_fn = nn.MSELoss(reduction='mean')
input_size = 1
output_size = 1
model = nn.Linear(input_size, output_size)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Now we can call step() method of the optimizer to train the model
# we can pass a batched dataset (such as train_dl)
for epoch in range(num_epochs):
    for x_batch, y_batch in train_dl:
        # 1. Generate predictions
        pred = model(x_batch)[:, 0]
        # 2. Calculate loss
        loss = loss_fn(pred, y_batch)
        # 3. Compute gradients
        loss.backward()
        # 4. Update parameters using gradients
        optimizer.step()
        # 5. Reset the gradients to zero
        optimizer.zero_grad()
    if epoch % log_epochs==0:
        print(f'Epoch {epoch} Loss {loss.item():.4f}')
print('Final Parameters:', model.weight.item(), model.bias.item())


# ## Building a multilayer perceptron for classifying flowers in the Iris dataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris['data']
y = iris['target']
# randomly select 100 samples (2/3) for training and 50 samples (1/3) for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1./3, random_state=1)

# Standardize features (mean centering and dividing by std) and create PyTorch Dataset for the training set
# and corresponding DataLoader
X_train_norm = (X_train - np.mean(X_train)) / np.std(X_train)
X_train_norm = torch.from_numpy(X_train_norm).float()
y_train = torch.from_numpy(y_train)
train_ds = TensorDataset(X_train_norm, y_train)
torch.manual_seed(1)
batch_size = 2
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

# Now we can use torch.nn module to build a model efficiently. We can stack a few layer and build an NN.
# Check available layers https://pytorch.org/docs/stable/nn.html. We will use Linear layer here, aka fully connected layer
# or dense layer, and best represented by f(w × x + b), where x is a tensor containing input features, w and b are weight
# matrix and bias vector and f is an activation function.

# Each layer in an NN receives its inputs from preceding layer; so its dimensionality (rank and shape)  is fixed.
# We want to define a model with two hidden layers. The first one receives an input of four features
# and projects them to 16 neurons.
# The second layer received the output of the previous layer (size of 16) and project them to three output neurons.
# Since we have three class labels, we can do as follows:

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = self.layer1(x)
        x = nn.Sigmoid()(x)
        x = self.layer2(x)
        x = nn.Softmax(dim=1)(x)
        return x
input_size = X_train_norm.shape[1] # columns = features = 4
hidden_size = 16
output_size = 3
model = Model(input_size, hidden_size, output_size)

# Note we used sigmoid activation function for the first layer and softmax activation for the last (output) layer)
# Softmax activation in the lsat layer is used to support multiclass classification since we have three class labels
# (which is why we have three neurons in the output layer).
# Next we specify loss function as cross-entropy loss and optimizer as Adam.
# (Adam optimizer is robust, gradient-based optimization method)

learning_rate = 0.001
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 100
loss_hist = [0] * num_epochs # create a list of 0s (done by multiplication with num_epochs) for loss history
accuracy_hist = [0] * num_epochs

for epoch in range(num_epochs):
    for x_batch, y_batch in train_dl:
        pred = model(x_batch)
        loss = loss_fn(pred, y_batch.long())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_hist[epoch] += loss.item()*y_batch.size(0)
        is_correct = (torch.argmax(pred, dim=1) == y_batch).float() # row for row find the index of the max value prediction
        accuracy_hist[epoch] += is_correct.sum()
    loss_hist[epoch] /= len(train_dl.dataset)
    accuracy_hist[epoch] /= len(train_dl.dataset)
# the loss_hist and accuracy_hist lists keeps the training loss and training accuracy
# after each epoch, we can use this to visualize learning curve as follows:
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(1, 2, 1)
ax.plot(loss_hist, lw=3)
ax.set_title('Training loss', size=15)
ax.set_xlabel('Epoch', size=15)
ax.tick_params(axis='both', which='major', labelsize=15)

ax = fig.add_subplot(1, 2, 2)
ax.plot(accuracy_hist, lw=3)
ax.set_title('Training accuracy', size=15)
ax.set_xlabel('Epoch', size=15)
ax.tick_params(axis='both', which='major', labelsize=15)
plt.tight_layout()
plt.show()


# ### Evaluating the trained model on the test dataset
X_test_norm = (X_test - np.mean(X_train)) / np.std(X_train)
X_test_norm = torch.from_numpy(X_test_norm).float()
y_test = torch.from_numpy(y_test)
pred_test = model(X_test_norm)
correct = (torch.argmax(pred_test, dim=1) == y_test).float()
accuracy = correct.mean()
print(f'Test Acc.: {accuracy:.4f}')


# ### Saving and reloading the trained model
path = 'iris_classifier.pt'
torch.save(model, path)

# Calling save(model) will save both model architecture and all learned parameters.
# We use pt or pth as common extension.

# verify model architecture by calling model.eval()
model_new = torch.load(path)
model_new.eval()

# evaluate new model reloaded on the test dataset to verify that the results are the same as before
pred_test = model_new(X_test_norm)
correct = (torch.argmax(pred_test, dim=1) == y_test).float()
accuracy = correct .mean()
print(f'Test Acc.: {accuracy:.4f}')

# if we only want to save learned parameters, we use save(model.state_dict())
##path = 'iris_classifier_state.pt'
##torch.save(model.state_dict(), path)

# to reload saved parameters, first construct the model as we did before,
# then feed the loaded parameters to the model
##model_new = Model(input_size, hidden_size, output_size)
##model_new.load_state_dict(torch.load(path))


# ## Choosing activation functions for multilayer neural networks
#

# ### Logistic function recap
# model for a two-dimensional data point, x and a model with weight coefficients assigned to w vector
X = np.array([1, 1.4, 2.5]) ## first value must be 1
w = np.array([0.4, 0.3, 0.5])
def net_input(X, w):
    return np.dot(X, w)
def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))
def logistic_activation(X, w):
    z = net_input(X, w)
    return logistic(z)

print(f'P(y=1|x) = {logistic_activation(X, w):.3f}')
# P(y=1|x) = 0.888
# we can interpret the value 0.888 as 88.8% probability that this  particular sample x, belongs to positive class

# W : array with shape = (n_output_units, n_hidden_units+1)
# note that the first column are the bias units

W = np.array([[1.1, 1.2, 0.8, 0.4],
              [0.2, 0.4, 1.0, 0.2],
              [0.6, 1.5, 1.2, 0.7]])

# A : data array with shape = (n_hidden_units + 1, n_samples)
# note that the first column of this array must be 1
A = np.array([[1, 0.1, 0.4, 0.6]])
Z = np.dot(W, A[0])
y_probas = logistic(Z)
print('Net Input: \n', Z)
print('Output Units:\n', y_probas)

# As we can see, the output result can't be interpreted as probabilities for a three-class
# problem. The reason for this is that they do not sum to 1. But this is not a big concern
# if we use our model to predict only the class labels and not the class membership probabilities.
# One way to predict the class label from the output units obtained earlier is to use max-value

y_class = np.argmax(Z, axis=0)
print('Predicted class label', y_class)


# ### Estimating class probabilities in multiclass classification via the softmax function
# softmax provides probability of each class, soft form of argmax function.
# it allows us to compute meaningful class probabilities in multiclass settings

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))
y_probas = softmax(Z)
print('Probabilities:\n', y_probas)
np.sum(y_probas) # now we can see the predicted class probablities sum to 1 as we would expect.

# it may help to think of the result of softmax function as a normalized output useful for obtaining
# meaningful class membership predictions in multiclass settings. Hence when we build a multiclass
# classification model in PyTorch, we can use torch.softmax() function to estimate the
# probabilties of each class membership for an input batch of examples.

# to use torch.softmax() activation function in PyTorch, we will convert Z to a tesnor in the following
# with an additional dimension reserved for the batch size:
torch.softmax(torch.from_numpy(Z), dim=0)


# ### Broadening the output spectrum using a hyperbolic tangent
# Another sigmoidal function used in hidden layers of artifical NNs is the hyperbolic tanget (aka tanh)
# which basically is  a rescaled version of logistic function. The advange over logistic function is that it has broader
# output spectrum ranging in the open interval (-1, 1) that can improve convergence of the backpropagation
# in contract logistic function returns and output signal ranging in the open interval (0, 1). For simple comparison:
def tanh(z):
    e_p = np.exp(z)
    e_m = np.exp(-z)
    return (e_p - e_m) / (e_p + e_m)
z = np.arange(-5, 5, 0.005)
log_act = logistic(z)
tanh_act = tanh(z)
plt.ylim([-1.5, 1.5])
plt.xlabel('Net input $z$')
plt.ylabel('Activation $\phi(z)$')
plt.axhline(1, color='black', linestyle=':')
plt.axhline(0.5, color='black', linestyle=':')
plt.axhline(0, color='black', linestyle=':')
plt.axhline(-0.5, color='black', linestyle=':')
plt.axhline(-1, color='black', linestyle=':')
plt.plot(z, tanh_act, linewidth=3, linestyle='--', label='tanh')
plt.plot(z, log_act, linewidth=3, label='logistic')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# As shown the shapes of the two sigmoidal curves look similar, but tanh function
# has double the output space of the logistic function.

# In practice we can use NumPy's tanh function rather than implementing it
# Alternatively when we build an NN model, we use torch.tanh(x) in PyTorch to achieve the same result
np.tanh(z)
torch.tanh(torch.from_numpy(z))

# Additionally logistic function is available in SciPy's special module
# Similarly we can use torch.sigmoid() function in PyTorch
from scipy.special import expit
expit(z)
torch.sigmoid(torch.from_numpy(z))

# Note that using torch.sigmoid(x) produces results that are equivalent to torch.
# nn.Sigmoid()(x), which we use earlier. torch.nn.Sigmoid is a class to which you
# can pass in parameters to construct an object in order to control the behavior. In contrast,
# torch.sigmoid is a function.

# ### Rectified linear unit activation (ReLU)
# Another activation function often  use in deep NNs
torch.relu(torch.from_numpy(z))



