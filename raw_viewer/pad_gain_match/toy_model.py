import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import scipy.optimize as opt
import cupy as cp

#build images
dim = 40
num_events = 10000
threshold = 0
counts_per_event=1000
sigma_min, sigma_max = 3,5
true_gains = np.random.normal(1, 0.05, (dim,dim))
sigma_to_edge = 0


device = 0

#build fake images
pad_images = []
for evt in range(num_events):
    sigma = np.random.uniform(sigma_min, sigma_max)
    point = np.random.uniform(sigma_to_edge*sigma, dim - sigma_to_edge*sigma, 2)
    image = np.zeros((dim, dim))
    image[*point.astype(int)] = counts_per_event
    image = scipy.ndimage.gaussian_filter(image, sigma, mode='reflect')/true_gains
    image[image<threshold] = 0
    pad_images.append(image)

plt.figure()
plt.imshow(image)
plt.colorbar()
print(np.sum(image))
plt.show()

with cp.cuda.Device(device):
    pad_images_cp = cp.array(pad_images)
    observed_counts_per_event = cp.sum(pad_images_cp)/num_events
    def obj_func(x):
        x = cp.array(x)
        gains = cp.reshape(x, (dim, dim))
        adc_counts_in_each_event = cp.einsum('ikj,kj', pad_images_cp, gains)/observed_counts_per_event
        return cp.asnumpy(cp.sqrt(cp.sum((adc_counts_in_each_event - 1)**2)/len(adc_counts_in_each_event))*2.355)

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

bins = np.linspace(np.min([gains, true_gains]), np.max([gains, true_gains]), 300)
plt.figure()
plt.hist(true_gains.flatten(), bins, label='true gains')
plt.hist(gains.flatten(), bins, label='fit gains', alpha=0.5)
plt.legend()

with cp.cuda.Device(device):
    def get_energy_per_event(x, cpimage):
        gains = cp.array(x)
        gains = cp.reshape(gains, (dim, dim))
        return cp.asnumpy(cp.einsum('ikj,kj', cpimage, gains))
    
counts_no_gain_match = get_energy_per_event(np.ones(dim*dim), pad_images_cp)
counts_w_gain_match = get_energy_per_event(res.x, pad_images_cp)
counts_w_true_gains = get_energy_per_event(cp.reshape(true_gains, (dim*dim,)), pad_images_cp)
plt.figure()
plt.title('energy spectrum: events use to fit pad gains')
bins = np.linspace(np.min([counts_no_gain_match, counts_w_gain_match]), np.max([counts_no_gain_match, counts_w_gain_match]), 300)
plt.hist(counts_no_gain_match, bins, label='counts without gain match')
plt.hist(counts_w_gain_match, bins, label='counts with gain match', alpha=0.5)
plt.legend()

#build a new set of fake images
pad_images2 = []
for evt in range(num_events):
    sigma = np.random.uniform(sigma_min, sigma_max)
    point = np.random.uniform(sigma_to_edge*sigma, dim - sigma_to_edge*sigma, 2)
    image = np.zeros((dim, dim))
    image[*point.astype(int)] = counts_per_event
    image = scipy.ndimage.gaussian_filter(image, sigma, mode='constant')/true_gains
    image[image<threshold] = 0
    pad_images2.append(image)
with cp.cuda.Device(device):
    pad_images_cp2 = cp.array(pad_images2)

counts_no_gain_match = get_energy_per_event(np.ones(dim*dim), pad_images_cp2)
counts_w_gain_match = get_energy_per_event(res.x, pad_images_cp2)
counts_w_true_gains = get_energy_per_event(cp.reshape(true_gains, (dim*dim,)), pad_images_cp2)
plt.figure()
plt.title('energy spectrum: new set of events')
bins = np.linspace(np.min([counts_no_gain_match, counts_w_gain_match]), np.max([counts_no_gain_match, counts_w_gain_match]), 300)
plt.hist(counts_no_gain_match, bins, label='counts without gain match')
plt.hist(counts_w_gain_match, bins, label='counts with gain match', alpha=0.5)
plt.hist(counts_w_true_gains, bins, label='counts with true gains', alpha=0.5)
plt.legend()

plt.show()
