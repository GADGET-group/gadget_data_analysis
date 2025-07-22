import pickle

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import scipy.optimize as opt
import cupy as cp

#image generation options
dim = 20
num_events = 10000
threshold = 0.5
counts_per_event=1000
sigma_min, sigma_max = 1.5,3
gain_mu, gain_sigma = 1, 0.0
true_gains = np.random.normal(gain_mu, gain_sigma, (dim,dim))
sigma_to_edge = 0
convolution_mode = 'reflect'
enable_edge_correction = True
load_res = False

if enable_edge_correction:
    save_name = 'dim%d_t%f_cpe%d_sig%f-%f_ste%f_gmu%fsig%f_edgecor.pkl'%(dim, threshold, counts_per_event, sigma_min, sigma_max,
                                                             sigma_to_edge, gain_mu, gain_sigma)
else:
    save_name = 'dim%d_t%f_cpe%d_sig%f-%f_ste%f_gmu%fsig%f.pkl'%(dim, threshold, counts_per_event, sigma_min, sigma_max,
                                                             sigma_to_edge, gain_mu, gain_sigma)


device = 0

def get_num_adj_pixels(images):
    to_return = []
    for image in images:
        image_copy =np.pad(image, 1) #pad image so pixels which would be just outsie the image, if the existed, are included
        image_copy[image_copy>0] = 1
        edge_kernel = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
        adj_image = scipy.ndimage.convolve(image_copy, edge_kernel)
        to_return.append(np.sum(adj_image<0)) #pixels ith negative values are edge pixels
    return to_return

#build fake images
if load_res:
    with open(save_name, 'rb') as f:
        pad_images = pickle.load(f)
        res = pickle.load(f)
else:
    pad_images = []
    for evt in range(num_events):
        sigma = np.random.uniform(sigma_min, sigma_max)
        point = np.random.uniform(sigma_to_edge*sigma, dim - sigma_to_edge*sigma, 2)
        image = np.zeros((dim, dim))
        image[*point.astype(int)] = counts_per_event
        image = scipy.ndimage.gaussian_filter(image, sigma, mode=convolution_mode)/true_gains
        image[image<threshold] = 0
        pad_images.append(image)


if False:
    plt.figure()
    plt.imshow(image)
    plt.colorbar()
    print(np.sum(image))
    plt.show()

with cp.cuda.Device(device):
    adj_pixel_counts_cp = cp.array(get_num_adj_pixels(pad_images))
    pad_images_cp = cp.array(pad_images)
    observed_counts_per_event = cp.sum(pad_images_cp)/num_events
    def obj_func(x):
        x = cp.array(x)
        if enable_edge_correction:
            gains = cp.reshape(x[:-1], (dim, dim))
        else:
            gains = cp.reshape(x, (dim, dim))
        adc_counts_in_each_event = cp.einsum('ikj,kj', pad_images_cp, gains)/observed_counts_per_event
        if enable_edge_correction:
            edge_corrections = cp.array(x[-1])*adj_pixel_counts_cp
            adc_counts_in_each_event += edge_corrections

        return cp.asnumpy(cp.sqrt(cp.sum((adc_counts_in_each_event - 1)**2)/len(adc_counts_in_each_event))*2.355)

    def callback(x):
        print(x[:10], x[-1], obj_func(x))
        print(np.mean(x), np.std(x), np.min(x), np.max(x))
    if not load_res:
        if enable_edge_correction:
            guess = np.ones(dim*dim+1)
            guess[-1]=0
        else:
            guess = np.ones(dim*dim)
        res = opt.minimize(obj_func, guess, callback = callback)
        with open(save_name, 'wb') as f:
            pickle.dump(pad_images, f)
            pickle.dump(res, f)

    
if enable_edge_correction:
    gains = cp.reshape(res.x[:-1], (dim, dim))
else:
    gains = cp.reshape(res.x, (dim, dim))
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
plt.hist(counts_no_gain_match, bins, label='counts without gain match, std=%f'%np.std(counts_no_gain_match))
plt.hist(counts_w_gain_match, bins, label='counts with gain match, std=%f'%np.std(counts_w_gain_match), alpha=0.5)
plt.hist(counts_w_true_gains, bins, label='counts with true gains, std=%f'%np.std(counts_w_true_gains), alpha=0.5)
#plt.yscale('log')
plt.legend()

#build a new set of fake images
pad_images2 = []
for evt in range(num_events):
    sigma = np.random.uniform(sigma_min, sigma_max)
    point = np.random.uniform(sigma_to_edge*sigma, dim - sigma_to_edge*sigma, 2)
    image = np.zeros((dim, dim))
    image[*point.astype(int)] = counts_per_event
    image = scipy.ndimage.gaussian_filter(image, sigma, mode=convolution_mode)/true_gains
    image[image<threshold] = 0
    pad_images2.append(image)
with cp.cuda.Device(device):
    pad_images_cp2 = cp.array(pad_images2)
    adj_pixel_counts_cp = cp.array(get_num_adj_pixels(pad_images))

counts_no_gain_match = get_energy_per_event(np.ones(dim*dim), pad_images_cp2)
counts_w_gain_match = get_energy_per_event(res.x, pad_images_cp2)
counts_w_true_gains = get_energy_per_event(cp.reshape(true_gains, (dim*dim,)), pad_images_cp2)
plt.figure()
plt.title('energy spectrum: new set of events')
bins = np.linspace(np.min([counts_no_gain_match, counts_w_gain_match]), np.max([counts_no_gain_match, counts_w_gain_match]), 300)
plt.hist(counts_no_gain_match, bins, label='counts without gain match, std=%f'%np.std(counts_no_gain_match))
plt.hist(counts_w_gain_match, bins, label='counts with gain match, std=%f'%np.std(counts_w_gain_match), alpha=0.5)
plt.hist(counts_w_true_gains, bins, label='counts with true gains, std=%f'%np.std(counts_w_true_gains), alpha=0.5)
#plt.yscale('log')
plt.legend()

#build a new set of fake images
pad_images3 = []
for evt in range(num_events):
    sigma = np.random.uniform(sigma_min, sigma_max)
    point = np.random.uniform(0, dim,2)
    image = np.zeros((dim, dim))
    image[*point.astype(int)] = counts_per_event
    image = scipy.ndimage.gaussian_filter(image, sigma, mode=convolution_mode)/true_gains
    image[image<threshold] = 0
    pad_images3.append(image)
with cp.cuda.Device(device):
    pad_images_cp3 = cp.array(pad_images3)

plt.figure()
plt.title('example new event')
plt.imshow(pad_images3[2])
plt.colorbar()

counts_no_gain_match = get_energy_per_event(np.ones(dim*dim), pad_images_cp3)
counts_w_gain_match = get_energy_per_event(res.x, pad_images_cp3)
counts_w_true_gains = get_energy_per_event(cp.reshape(true_gains, (dim*dim,)), pad_images_cp3)
plt.figure()
plt.title('energy spectrum: new set of events')
bins = np.linspace(np.min([counts_no_gain_match, counts_w_gain_match]), np.max([counts_no_gain_match, counts_w_gain_match]), 300)
plt.hist(counts_no_gain_match, bins, label='counts without gain match, std=%f'%np.std(counts_no_gain_match))
plt.hist(counts_w_gain_match, bins, label='counts with gain match, std=%f'%np.std(counts_w_gain_match), alpha=0.5)
plt.hist(counts_w_true_gains, bins, label='counts with true gains, std=%f'%np.std(counts_w_true_gains), alpha=0.5)
#plt.yscale('log')
plt.legend()
plt.show()

dim_min_max = [1,5]
samples = 1000
rec_dims = []
for i in range(samples):
    center = np.random.uniform(0, dim, 2)
    w_nominal = np.random.uniform(*dim_min_max)
    h_nominal = np.random.uniform(*dim_min_max)
