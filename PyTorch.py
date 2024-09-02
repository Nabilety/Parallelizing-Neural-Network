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
