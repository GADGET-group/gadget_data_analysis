import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import scipy.optimize as opt

#build images
dim = 40
num_events = 100000
threshold = 6
counts_per_event=1000
sigma_min, sigma_max = 3,5
true_gains = np.random.normal(1, 0.05, (dim,dim))
sigma_to_edge = 2

#build fake images
pad_images = []
for evt in range(num_events):
    sigma = np.random.uniform(sigma_min, sigma_max)
    point = np.random.uniform(sigma_to_edge*sigma, dim - sigma_to_edge*sigma, 2)
    image = np.zeros((dim, dim))
    image[*point.astype(int)] = counts_per_event
    image = scipy.ndimage.gaussian_filter(image, sigma, mode='constant')*true_gains
    image[image<threshold] = 0
    pad_images.append(image)

plt.figure()
plt.imshow(image)
plt.colorbar()
plt.show()

def obj_func(x):
    gains = np.reshape(x, (dim, dim))
    adc_counts_in_each_event = np.einsum('ikj,kj', pad_images, gains)/counts_per_event
    return np.sqrt(np.sum((adc_counts_in_each_event - 1)**2)/len(adc_counts_in_each_event))*2.355

def callback(x):
    print(x, obj_func(x))
    print(np.mean(x), np.std(x), np.min(x), np.max(x))

res = opt.minimize(obj_func, np.ones(dim*dim), callback = callback)
gains = np.reshape(res.x, (dim, dim))
print(res)

plt.figure()
plt.title('sum of events')
plt.imshow(np.sum(pad_images, axis=0))
plt.colorbar()

plt.figure()
plt.title('true gains')
plt.imshow(true_gains)
plt.colorbar()

plt.figure()
plt.title('fit gains')
plt.imshow(gains)
plt.colorbar()

plt.figure()
plt.title('gain residuals (fit - truth)')
plt.imshow(gains - true_gains)
plt.colorbar()
plt.show()
