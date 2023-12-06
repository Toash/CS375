import numpy as np
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
import torch


X = np.empty((0, 784))
dataset = np.load("./Data/full_numpy_bitmap_pizza.npy")
X = np.concatenate((X, dataset[:1]), axis=0)
#print(X)

#   Normalize
X = ((X / 255.0) - 0.5) * 2
X = np.reshape(X, (X.shape[0], 1, 28, 28))  #   1 color channel for 28x28 image.
#print(X)
X = torch.from_numpy(X).float() 

transform = transforms.Compose([
        transforms.RandomAffine(degrees=30, translate=(.2, .2), scale=(.8, 1.2), fill=-1),
        transforms.RandomCrop(28, pad_if_needed=True)
])

# np reshape implicitly converts torch tensor to np array
orig = np.reshape(X,(28,28))
trans = transform(X)
trans = np.reshape(trans,(28,28))


plt.subplot(1, 2, 1)
plt.imshow(orig, cmap="gray")
plt.title("Original")

plt.subplot(1, 2, 2)
plt.imshow(trans, cmap="gray")
plt.title("Transformed")
plt.show()
