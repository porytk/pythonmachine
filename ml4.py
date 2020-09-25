from scipy.io import loadmat
import matplotlib.pyplot as plt
mnist_raw = loadmat("mnist-original.mat")
# print(mnist_raw) #alldata
mnist = {
    "data": mnist_raw["data"].T,
    "target": mnist_raw["label"][0]
}
x = mnist["data"]
y = mnist["target"]
# x,y=mnist["data"],mnist["target"]
# print(mnist["data"].shape)
number = x[10000]
number_image = number.reshape(28, 28)
print(x.shape)
# print(y) number all
print(y[10000])
plt.imshow(
    number_image,
    cmap=plt.cm.binary,
    interpolation="nearest")
plt.show()
