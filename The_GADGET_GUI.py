#!/usr/bin/env python
# coding: utf-8
#ORIGINAL GADGET GUI WRITTEN BY TYLER WHEELER

############################################################### File Generator
###############################################################

def generate_files(run_num, length, ic, pads, eps, samps, poly):
	import numpy as np
	import h5py
	import math
	from sklearn.decomposition import PCA
	from pca import pca
	import pandas as pd
	import matplotlib.pyplot as plt
	from matplotlib.colors import LinearSegmentedColormap
	from tqdm import tqdm
	import peakutils
	import os, sys
	from BaselineRemoval import BaselineRemoval
	from sklearn.cluster import DBSCAN
	from scipy.signal import find_peaks, savgol_filter



	# In[2]:
	class HiddenPrints:
		def __enter__(self):
			self._original_stdout = sys.stdout
			sys.stdout = open(os.devnull, 'w')

		def __exit__(self, exc_type, exc_val, exc_tb):
			sys.stdout.close()
			sys.stdout = self._original_stdout

	def smooth_trace(trace, window_length=15, polyorder=3):
		smoothed_trace = savgol_filter(trace, window_length, polyorder)
		return smoothed_trace

	def remove_noise(trace, threshold_ratio=0.1):
		threshold = threshold_ratio * np.max(np.abs(trace))
		trace[np.abs(trace) < threshold] = 0

		# Remove negative values
		trace[trace < 0] = 0

		# Find the index of the maximum value in the trace
		max_idx = np.argmax(trace)

		# Zero out bins to the left of the maximum value if a zero bin is encountered
		for i in range(max_idx - 1, -1, -1):
			if trace[i] == 0:
				trace[:i] = 0
				break

		# Zero out bins to the right of the maximum value if a zero bin is encountered
		for i in range(max_idx + 1, len(trace)):
			if trace[i] == 0:
				trace[i:] = 0
				break

		return trace



	def remove_outliers(xset, yset, zset, eset, pads):
		"""
		Uses DBSCAN to find and remove outliers in 3D data
		"""

		data = np.array([xset.T, yset.T, zset.T]).T
		DBSCAN_cluster = DBSCAN(eps=eps, min_samples=samps).fit(data)
		del data
	
		if all(element == -1 for element in DBSCAN_cluster.labels_):
			veto = True
		else:
			# Identify largest clusters
			labels = DBSCAN_cluster.labels_
			unique_labels = set(labels)
			if -1 in unique_labels:
				unique_labels.remove(-1)

			# Find the two largest clusters
			largest_clusters = sorted(unique_labels, key=lambda x: np.sum(labels == x), reverse=True)[:2]

			# Relabel non-main-cluster points as outliers
			for cluster_label in unique_labels:
				if cluster_label not in largest_clusters:
					labels[labels == cluster_label] = -1

			# Remove outlier points
			out_of_cluster_index = np.where(labels == -1)
			rev = out_of_cluster_index[0][::-1]
			for i in rev:
				xset = np.delete(xset, i)
				yset = np.delete(yset, i)
				zset = np.delete(zset, i)
				eset = np.delete(eset, i)

			if len(xset) <= pads:
				veto = True
			else:
				veto = False

		return xset, yset, zset, eset, veto

	"""
	def track_len(xset, yset, zset):

		#Uses PCA to find the length of a track
		
		veto_on_length = False

		# Form data matrix
		data = np.concatenate((xset[:, np.newaxis], 
				   yset[:, np.newaxis], 
				   zset[:, np.newaxis]), 
				   axis=1)

		# Use PCA to find track length
		pca = PCA(n_components=2)
		principalComponents = pca.fit(data)
		principalComponents = pca.transform(data)
		principalDf = pd.DataFrame(data = principalComponents
		     , columns = ['principal component 1', 'principal component 2'])
		calibration_factor = 1.4 
		track_len = calibration_factor*2.35*principalDf.std()[0]
		#track_len = 2.35*principalDf.std()[0]
		if track_len > length:
			veto_on_length = True

		return track_len, veto_on_length
	"""
	def track_len(xset, yset, zset):
		"""
		Uses PCA to find the length of a track
		"""
		veto_on_length = False

		# Form data matrix
		data = np.concatenate((xset[:, np.newaxis], 
		       yset[:, np.newaxis], 
		       zset[:, np.newaxis]), 
		       axis=1)

		# Use PCA to find track length
		pca = PCA(n_components=2)
		principalComponents = pca.fit(data)
		principalComponents = pca.transform(data)
		principalDf = pd.DataFrame(data = principalComponents
		 , columns = ['principal component 1', 'principal component 2'])
		calibration_factor = 1.4 

		# Call track_angle to get the angle of the track
		angle_deg = track_angle(xset, yset, zset)

		# Calculate the scale factor based on the angle
		#scale_factor = get_scale_factor(angle_deg)
		#scale_factor_trace = scale_factor * 1.5 

		# Apply the scale factor to the track length
		# track_len = scale_factor * calibration_factor * 2.35 * principalDf.std()[0]
		track_len = calibration_factor * 2.35 * principalDf.std()[0]

		if track_len > length:
			veto_on_length = True

		return track_len, veto_on_length, angle_deg



	def track_angle(xset, yset, zset):
		"""
		Fits 3D track, and determines angle wrt pad plane
		"""
		# Form data matrix
		data = np.concatenate((xset[:, np.newaxis], 
				   yset[:, np.newaxis], 
				   zset[:, np.newaxis]), 
				   axis=1)

		# Fit regression line
		line_fit = Line.best_fit(data)

		# Find angle between the vector of the fit line and a vector normal to the xy-plane (pad plane)
		v = np.array([line_fit.vector]).T   # fit line vector
		n = np.array(([[0, 0, 1]])).T       # Vector normal to xy-plane
		dot = np.dot(n.T, v)[0][0]          # Note that both vectors already have a magnitude of 1

		# Clamp the dot variable to be within the valid range
		dot = max(-1.0, min(1.0, dot))

		theta = math.acos(dot)
		track_angle_rad = (math.pi/2 - theta) 
		track_angle_deg = track_angle_rad * (180 / np.pi)

		# Angle should always be less than 90 deg
		if track_angle_deg < 0:
			track_angle_deg = 180 + track_angle_deg 
		if track_angle_deg > 90:
			track_angle_deg = 180 - track_angle_deg

		return track_angle_deg

	def get_scale_factor(angle, angle_min=40, angle_max=90, scale_min=1, scale_max=1.3):
		if angle < angle_min:
			return scale_min
		elif angle > angle_max:
			return scale_max
		else:
			return scale_min + (scale_max - scale_min) * (angle - angle_min) / (angle_max - angle_min)


	def main(h5file, pads, ic):
		"""
		This functions does the following: 
		- Converts h5 files into ndarrays. 
		- Removes outliers.
		- Calls PCA to return track length.
		- Sums mesh signal to return energy.
		"""
		# Converts h5 files into ndarrays, and output each event dataset as a separte list
		num_events = int(len(np.array(h5file['clouds'])))

		len_list = []
		good_events = []
		tot_energy = []
		trace_list = []
		xHit_list = []
		yHit_list = []
		zHit_list = []
		eHit_list = []
		angle_list = []

		cloud_missing = 0
		skipped_events = 0
		veto_events = 0

		# Veto in "junk" region of plot (low energy, high range)
		# Define veto region
		y = np.array([20, 40, 60])
		x = np.array([100000, 150000, 200000])
		slope, intercept = np.polyfit(x, y, deg=1)

		pbar = tqdm(total=num_events+1)
		for i in range(1, num_events+1):

			# Make copy of cloud datasets
			str_cloud = f"evt{i}_cloud"
			try:
				cloud = np.array(h5file['clouds'][str_cloud])
			except:
				cloud_missing += 1
				pbar.update(n=1)
				continue

			# Make copy of datasets
			cloud_x = cloud[:,0]
			cloud_y = cloud[:,1]
			cloud_z = cloud[:,2]
			#cloud_z = cloud[:,2] - np.min(cloud[:, 2])
			cloud_e = cloud[:,3]
			del cloud

			# Apply veto condition
			R = 36                           # Radius of the pad plane
			r = np.sqrt(cloud_x**2 + cloud_y**2)
			statements = np.greater(r, R)    # Check if any point lies outside of R

			if np.any(statements) == True:
				veto_events += 1
				pbar.update(n=1)
				continue
			
			# Apply pad threshold
			x = (35 + cloud_x) * 2 + 42
			y = 145 - (35 + cloud_y) * 2
			xy_tuples = np.column_stack((x, y))
			unique_xy_tuples = set(map(tuple, xy_tuples))
			num_unique_tuples = len(unique_xy_tuples)

			if num_unique_tuples <= pads:
				skipped_events += 1
				pbar.update(n=1)
				continue

			"""
			# Call remove_outliers to get dataset w/ outliers removed
			cloud_x, cloud_y, cloud_z, cloud_e, veto = remove_outliers(cloud_x, cloud_y, cloud_z, cloud_e, pads)
			if veto == True:
				skipped_events += 1
				pbar.update(n=1)
				continue
			"""
			# Move track next to pad plane for 3D view and scale by appropriate factor
			cloud_z = (cloud_z  - np.min(cloud_z ))*1.45 # Have also used 1.92
			#cloud_z = (cloud_z  - np.min(cloud_z ))

			# Call track_len() to create lists of all track lengths
			length, veto_on_length, angle = track_len(cloud_x, cloud_y, cloud_z)
			if veto_on_length == True:
				veto_events += 1
				pbar.update(n=1)
				continue 

			str_trace = f"evt{i}_data"
			data_trace = np.array(h5file['get'][str_trace])
			# pad_nums = data_trace[:,4]			 

			trace = np.sum(data_trace[:, -512:], axis=0)
			del data_trace


			max_val = np.argmax(trace)
			low_bound = max_val - 75
			if low_bound < 0:
				low_bound = 5
			upper_bound = max_val + 75
			if upper_bound > 512:
				upper_bound = 506
			trace = trace[low_bound:upper_bound]

			# Smooth trace
			trace = smooth_trace(trace)

			# Subtract background and fit trace
			polynomial_degree=poly 
			baseObj=BaselineRemoval(trace)
			trace=baseObj.IModPoly(polynomial_degree)

			# Remove noise, negative values, and zero consecutive bins
			trace = remove_noise(trace, threshold_ratio=0.01)
	

			# Here you can apply the scale factor to the total energy
			scaled_energy = np.sum(trace)

			if scaled_energy > ic:
				veto_events += 1
				pbar.update(n=1)
				continue

			"""
			# Check to see if point is in "junk" region
			x1 = scaled_energy
			y1 = length
			y_line = slope * x1 + intercept
			if y1 > y_line and y1 > 20:
				veto_events += 1
				pbar.update(n=1)
				continue
			"""

			# Call track_angle to create list of all track angles
			angle_list.append(angle)

			# Append all lists
			len_list.append(length)
			tot_energy.append(scaled_energy)
			trace_list.append(trace)
			xHit_list.append(cloud_x)
			yHit_list.append(cloud_y)
			zHit_list.append(cloud_z)
			eHit_list.append(cloud_e)

			# Track original event number of good events
			good_events.append(i)
			pbar.update(n=1)

		print('Starting # of Events:', num_events)
		print('Events Below Threshold:', skipped_events)
		print('Vetoed Events:', veto_events)
		print('Events Missing Cloud:', cloud_missing)
		print('Final # of Good Events:', len(good_events))

		return (tot_energy, skipped_events, veto_events, good_events, len_list, trace_list, xHit_list, yHit_list, zHit_list, eHit_list, angle_list)

	

	#str_file = f"/mnt/rawdata/e21072/h5/run_{run_num}.h5"
	str_file = f"/mnt/analysis/e21072/h5test/run_{run_num}.h5"
	f = h5py.File(str_file, 'r')
	(tot_energy, skipped_events, veto_events, good_events, len_list, trace_list, xHit_list, yHit_list, zHit_list, eHit_list, angle_list) = main(h5file=f, pads=pads, ic=ic)

	# Save Arrays
	print(f"DIRECTORY: /mnt/analysis/e21072/h5test/run_{run_num}/len{length}_ic{ic}_pads{pads}_eps{eps}_samps{samps}_poly{poly}")
	sub_path = f"/mnt/analysis/e21072/h5test/run_{run_num}/len{length}_ic{ic}_pads{pads}_eps{eps}_samps{samps}_poly{poly}/tot_energy"
	np.save(sub_path, tot_energy, allow_pickle=True)

	sub_path = f"/mnt/analysis/e21072/h5test/run_{run_num}/len{length}_ic{ic}_pads{pads}_eps{eps}_samps{samps}_poly{poly}/skipped_events"
	np.save(sub_path, skipped_events, allow_pickle=True)

	sub_path = f"/mnt/analysis/e21072/h5test/run_{run_num}/len{length}_ic{ic}_pads{pads}_eps{eps}_samps{samps}_poly{poly}/veto_events"
	np.save(sub_path, veto_events, allow_pickle=True)

	sub_path = f"/mnt/analysis/e21072/h5test/run_{run_num}/len{length}_ic{ic}_pads{pads}_eps{eps}_samps{samps}_poly{poly}/good_events"
	np.save(sub_path, good_events, allow_pickle=True)

	sub_path = f"/mnt/analysis/e21072/h5test/run_{run_num}/len{length}_ic{ic}_pads{pads}_eps{eps}_samps{samps}_poly{poly}/len_list"
	np.save(sub_path, len_list, allow_pickle=True)

	sub_path = f"/mnt/analysis/e21072/h5test/run_{run_num}/len{length}_ic{ic}_pads{pads}_eps{eps}_samps{samps}_poly{poly}/trace_list"
	np.save(sub_path, trace_list, allow_pickle=True)

	sub_path = f"/mnt/analysis/e21072/h5test/run_{run_num}/len{length}_ic{ic}_pads{pads}_eps{eps}_samps{samps}_poly{poly}/xHit_list"
	np.save(sub_path, xHit_list, allow_pickle=True)

	sub_path = f"/mnt/analysis/e21072/h5test/run_{run_num}/len{length}_ic{ic}_pads{pads}_eps{eps}_samps{samps}_poly{poly}/yHit_list"
	np.save(sub_path, yHit_list, allow_pickle=True)

	sub_path = f"/mnt/analysis/e21072/h5test/run_{run_num}/len{length}_ic{ic}_pads{pads}_eps{eps}_samps{samps}_poly{poly}/zHit_list"
	np.save(sub_path, zHit_list, allow_pickle=True)

	sub_path = f"/mnt/analysis/e21072/h5test/run_{run_num}/len{length}_ic{ic}_pads{pads}_eps{eps}_samps{samps}_poly{poly}/eHit_list"
	np.save(sub_path, eHit_list, allow_pickle=True)

	sub_path = f"/mnt/analysis/e21072/h5test/run_{run_num}/len{length}_ic{ic}_pads{pads}_eps{eps}_samps{samps}_poly{poly}/angle_list"
	np.save(sub_path, angle_list, allow_pickle=True)

	#Delete arrays
	del tot_energy
	del skipped_events
	del veto_events
	del good_events
	del len_list
	del trace_list
	del xHit_list
	del yHit_list
	del zHit_list
	del eHit_list
	del angle_list
	
	print('Files Saved. Switch to Existing Files.')

	return 



############################################################## Range vs Energy
##############################################################	

def RvE():

	# Get good_events list to return original index of events
	def plot_spectrum():
		import matplotlib.colors as colors

		# (energy, int_charge)
		calib_point_1 = (0.806, 156745)
		calib_point_2 = (1.679, 320842)
		#calib_point_1 = (0.806, 157600)
		#calib_point_2 = (1.679, 275300)
		#calib_point_1 = (0.303, 84672)
		#calib_point_2 = (2.150, 374439)
		#calib_point_1 = (0.806, 157700)
		#calib_point_2 = (1.679, 308200)


		# Convert calibration points to MeV
		energy_1, channel_1 = calib_point_1
		energy_2, channel_2 = calib_point_2
		energy_scale_factor = (energy_2 - energy_1) / (channel_2 - channel_1)
		energy_offset = energy_1 - energy_scale_factor * channel_1

		# Convert tot_energy to MeV
		tot_energy_MeV = np.array(tot_energy) * energy_scale_factor + energy_offset

		num_bins = int(entry_bins_RvE.get())

		print('Plotting')
	
		# Plot 2D histogram of range vs energy
		plt.rcParams['figure.figsize'] = [10, 10]
		plt.xlabel('Energy (MeV)', fontdict={'fontsize': 20})
		plt.ylabel('Range (mm)', fontdict={'fontsize': 20})
		plt.title(f'Range vs Energy | Bins = {num_bins}', fontdict={'fontsize': 20})
		tot_energy_temp = np.concatenate(([0], tot_energy_MeV))
		len_list_temp = np.concatenate(([0], len_list))

		#hist_values, x_edges, y_edges, img = plt.hist2d(tot_energy_temp, len_list_temp, (num_bins, num_bins), cmap=plt.cm.jet)
		# Use a logarithmic color scale for the histogram
		plt.hist2d(tot_energy_temp, len_list_temp, (num_bins, num_bins), cmap=plt.cm.jet, norm=colors.LogNorm())

		plt.colorbar()

		# Change the background color to dark blue
		plt.gca().set_facecolor('darkblue')

		########### We are in log scale

		plot = plt.show()
		return plot

		"""
		# Find the appropriate range for the y-axis of the line plot
		y_min = np.min(y_edges)
		y_max = np.max(y_edges)

		# Plot the line given by the equation -12.9 + 11.5x + 10.7x^2 
		x_values = np.linspace(1.27, 2.21, 1000) # Total energy range, 2 sigma
		y_values = -12.9 + 11.5 * x_values + 10.7 * x_values**2

		# Rescale the y-axis of the line plot to match the range of the histogram
		y_values_rescaled = (y_values - y_min) / (y_max - y_min) * (np.max(len_list_temp) - np.min(len_list_temp)) + np.min(len_list_temp)

		# Plot the second line that goes from (x=1.49, y=31.79) to (x=1.96, y=60.19)
		y2_start, y2_end = 26.46, 83.57 # +sigma range values
		y2_values = np.linspace(y2_start, y2_end, 1000)
		plt.plot(x_values, y2_values, 'b-', linewidth=2, label="Uppper Bound")

		# Plot the third line that goes from (x=1.49, y=24.29) to (x=1.96, y=41.39)
		y3_start, y3_end = 11.46, 45.97 # -sigma range values
		y3_values = np.linspace(y3_start, y3_end, 1000)
		plt.plot(x_values, y3_values, 'g-', linewidth=2, label="Lower Bound")

		# Define the coordinates of the trapezoid corners
		x_trapezoid = [x_values[0], x_values[-1], x_values[-1], x_values[0]]
		y_trapezoid = [y3_values[0], y3_values[-1], y2_values[-1], y2_values[0]]

		# Plot the trapezoid
		plt.fill(x_trapezoid, y_trapezoid, color='gray', alpha=0.9, label='Region of Interest (2 std)')

		plt.plot(x_values, y_values_rescaled, 'r-', linewidth=2, label="y = -12.9 + 11.5x + 10.7x^2")
		plt.legend()
		"""

	def plot_spectrum_point():
		calib_point_1 = (0.806, 156745)
		calib_point_2 = (1.679, 320842)

		energy_1, channel_1 = calib_point_1
		energy_2, channel_2 = calib_point_2
		energy_scale_factor = (energy_2 - energy_1) / (channel_2 - channel_1)
		energy_offset = energy_1 - energy_scale_factor * channel_1

		tot_energy_MeV = np.array(tot_energy) * energy_scale_factor + energy_offset

		num_bins = int(entry_bins_RvE.get())

		orig_num = int(entry_event_num.get())
		event_num = np.where(good_events == orig_num)[0][0]

		print('Plotting')
		plt.rcParams['figure.figsize'] = [10, 10]
		plt.xlabel('Energy (MeV)', fontdict={'fontsize': 20})
		plt.ylabel('Range (mm)', fontdict={'fontsize': 20})
		str_title = f"Range vs Energy w/ Point {good_events[event_num]}"
		plt.title(str_title, fontdict={'fontsize': 20})
		tot_energy_temp = np.concatenate(([0], tot_energy_MeV))
		len_list_temp = np.concatenate(([0], len_list))
		plt.hist2d(tot_energy_temp, len_list_temp, (num_bins, num_bins), cmap=plt.cm.jet)

		point, = plt.plot(tot_energy_MeV[event_num], len_list[event_num], 'ro', picker=5)  # 5 points tolerance
		annotation = plt.annotate(f"Evt#: {orig_num}", (tot_energy_MeV[event_num], len_list[event_num]),
                          textcoords="offset points", xytext=(-15,7), ha='center', 
                          fontsize=10, color='black',
                          bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", edgecolor="black"))

		annotation.set_visible(False)

		def on_pick(event):
			if event.artist != point: return
			annotation.set_visible(not annotation.get_visible())
			plt.draw()

		plt.gcf().canvas.mpl_connect('pick_event', on_pick)
		plt.show()


	def temp_text4(e):
		entry_low.delete(0,"end")

	def temp_text5(e):
		entry_high.delete(0,"end")


	def temp_text10(e):
		entry_event_num.delete(0,"end")


	def switch_RvE():
		global is_on_RvE

		# Determine is on or off
		if is_on_RvE:
			is_on_RvE = False
			entry_bins_eng_spec.destroy()
			button_spec.destroy()
			button_RvE.configure(fg='red')

			
		else:
			is_on_RvE = True
			button_RvE.configure(fg='green')
			open_RvE()
		
	
	def prev_cut():
		from tqdm import tqdm 
		import glob
		# All files and directories ending with .txt and that don't begin with a dot:
		prev_cut_path = os.path.join(sub_mymainpath, "*.jpg")
		global init_image_list
		init_image_list = glob.glob(prev_cut_path)
		image_list = []
		for i in range(len(init_image_list)):
			image_list.append(ImageTk.PhotoImage(Image.open(init_image_list[i])))


		newWindow = Toplevel(root)
		newWindow.title('Image Viewer')
		newWindow.geometry("1000x1000")

		def update_title(image_index, init_image_list):
			dir_name = init_image_list[image_index][:-4]
			newWindow.title(f'Image Viewer - {dir_name}')


		update_title(0, init_image_list)

		global my_label
		my_label = Label(newWindow, image=image_list[0])
		my_label.place(x=0, y=0)
		
		global image_index 
		image_index = 0

		def forward(image_number):
			global my_label
			global button_forward
			global button_back
			image_index = image_number - 1
			update_title(image_number - 1, init_image_list)

			my_label.place_forget()
			my_label = Label(newWindow, image=image_list[image_number-1])
			button_forward = Button(newWindow, text=">>", command=lambda: forward(image_number+1))
			button_exit = Button(newWindow, text="Exit Program", command=newWindow.destroy)
			button_back = Button(newWindow, text="<<", command=lambda: back(image_number-1))
			button_select = Button(newWindow, text="Select Cut", command=lambda: select_cut(image_index, init_image_list))
			button_h5 = Button(newWindow, text="Create H5 File", command=lambda: create_h5(image_index, init_image_list))
			
			if image_number == len(init_image_list):
				button_forward = Button(newWindow, text=">>", state=DISABLED)

			my_label.place(x=0, y=0)
			button_back.place(x=350, y=40)
			button_exit.place(x=470, y=80)
			button_select.place(x=475, y=40)
			button_forward.place(x=635, y=40)
			button_h5.place(x=470, y=950)

		def back(image_number):
			global my_label
			global button_forward
			global button_back
			image_index = image_number - 1
			update_title(image_number - 1, init_image_list)

			my_label.place_forget()
			my_label = Label(newWindow, image=image_list[image_number-1])
			button_forward = Button(newWindow, text=">>", command=lambda: forward(image_number+1))
			button_exit = Button(newWindow, text="Exit Program", command=newWindow.destroy)
			button_back = Button(newWindow, text="<<", command=lambda: back(image_number-1))
			button_select = Button(newWindow, text="Select Cut", command=lambda: select_cut(image_index, init_image_list))
			button_h5 = Button(newWindow, text="Create H5 File", command=lambda: create_h5(image_index, init_image_list))

			if image_number == 1:
				button_back = Button(newWindow, text="<<", state=DISABLED)

	
			my_label.place(x=0, y=0)
			button_back.place(x=350, y=40)
			button_exit.place(x=470, y=80)
			button_select.place(x=475, y=40)
			button_forward.place(x=635, y=40)
			button_h5.place(x=470, y=950)


		def create_h5(image_index, init_image_list):
			import pickle
			import h5py
			from tqdm import tqdm
			
			print('CUT DIRECTORY \n',init_image_list[image_index][:-4])
			dir_select = init_image_list[image_index][:-4]
			newWindow.destroy()

			cut_indicies_str = "cut_indices_H5list.pkl"
			cut_indicies_path = os.path.join(dir_select, cut_indicies_str)

			with open(cut_indicies_path, "rb") as file:
    				cut_indices_H5list = pickle.load(file)

			str_file = f"/mnt/analysis/e21072/h5test/run_{run_num}.h5"

			# Open the original h5 file
			with h5py.File(str_file, 'r') as original:
				# Get the Event and Trace groups
				cloud_group = original['clouds']
				trace_group = original['get']

				# Create a new h5 file
				h5_str = f"h5_cut.h5"
				h5_path = os.path.join(dir_select, h5_str)
				with h5py.File(h5_path, 'w') as new:
					# Create the Event and Trace groups in the new file
					new_cloud_group = new.create_group('clouds')
					new_trace_group = new.create_group('get')

					# Loop over the indices and extract the corresponding datasets
					pbar = tqdm(total=len(cut_indices_H5list))
					for i in cut_indices_H5list:
						cloud_dataset = cloud_group[f'evt{i}_cloud']
						trace_dataset = trace_group[f'evt{i}_data']

						# Copy the datasets to the new file
						new_cloud_group.create_dataset(f'evt{i}_cloud', data=cloud_dataset)
						new_trace_group.create_dataset(f'evt{i}_data', data=trace_dataset)
						pbar.update(n=1)


		def select_cut(image_index, init_image_list):
			from PIL import Image, ImageTk, ImageOps
			print('CUT DIRECTORY \n', init_image_list[image_index][:-4])
			dir_select = init_image_list[image_index][:-4]
			newWindow.destroy()

			prev_cut_path = os.path.join(dir_select, "*.jpg")
			global cut_image_list
			cut_image_list = glob.glob(prev_cut_path)

			image_list = []
			pbar = tqdm(total=len(cut_image_list))
			for i in range(len(cut_image_list)):
				image_list.append(ImageTk.PhotoImage(Image.open(cut_image_list[i])))
				pbar.update(n=1)

			newWindow2 = Toplevel(root)
			newWindow2.title('Image Viewer')
			newWindow2.geometry("695x695")

			global my_label
			my_label = Label(newWindow2, image=image_list[0])
			my_label.place(x=0, y=0)

			grid_mode = False

			current_image_index = 0  # Add this line to keep track of the current image index

			def go_to_image(event_num):
				global good_events
				event_idx = np.where(good_events == event_num)[0][0]
				image_name = f'image_{event_idx}.jpg'

				for idx, img_path in enumerate(cut_image_list):
					if image_name in img_path:
						update_single(idx)
						break
					
			def change_mode():
				nonlocal grid_mode
				grid_mode = not grid_mode
				if grid_mode:
					button_grid.config(text="Single Image")
					newWindow2.geometry("1025x995")  # Increase window size for 3x3 grid view
					update_grid(current_image_index)  # Pass current_image_index
				else:
					button_grid.config(text="3x3 Grid")
					newWindow2.geometry("695x695")
					update_single(current_image_index)

			def update_single(index):
				nonlocal grid_mode, current_image_index  # Add current_image_index to nonlocal variables
				current_image_index = index  # Update current_image_index with the new index
				if not grid_mode:
					for label in newWindow2.place_slaves():  # Destroy grid labels
                				if label != my_label and label not in (button_back, button_exit, button_forward, button_grid):
                    					label.destroy()
					my_label.config(image=image_list[index])
					button_forward.config(command=lambda: update_single(index + 1))
					button_back.config(command=lambda: update_single(index - 1))

					if index == 0:
						button_back.config(state=DISABLED)
					else:
						button_back.config(state=NORMAL)

					if index == len(cut_image_list) - 1:
						button_forward.config(state=DISABLED)
					else:
						button_forward.config(state=NORMAL)

					print(f'{index} of {len(cut_image_list)}')
					print('Cut Index:', os.path.basename(cut_image_list[index]))

					# Set button placement for single image viewer
					button_back.place(x=200, y=10)
					button_forward.place(x=485, y=10)
					button_grid.place(x=330, y=10)

					# Create and place the entry field and "Go to Image" button
					entry_field = Entry(newWindow2, width=10)
					entry_field.place(x=330, y=635)
					go_to_image_button = Button(newWindow2, text="Go to Image", command=lambda: go_to_image(int(entry_field.get())))
					go_to_image_button.place(x=325, y=665)

			def update_grid(index):
				nonlocal grid_mode, current_image_index
				current_image_index = index
				if grid_mode:
					my_label.config(image="")
					for label in newWindow2.place_slaves():
						if label != my_label and label not in (button_back, button_exit, button_forward, button_grid):
                    					label.destroy()
					grid_images = []  # Add this line to store grid_image objects
					
					for i in range(3):
						for j in range(3):
							img_idx = index + i * 3 + j
							if img_idx < len(image_list):
								im = Image.open(cut_image_list[img_idx])
								im.thumbnail((420, 340), Image.Resampling.LANCZOS)  # Resize images to 360x270 pixels
								grid_image = ImageTk.PhotoImage(im)
								grid_label = Label(newWindow2, image=grid_image)
								grid_label.image = grid_image
								grid_label.place(x=320 * j+ 22 * j, y=40+240 * i + 70*i)  # Adjust image placement in the 3x3 grid view by adjusting final values infront of i/j
								grid_images.append(grid_image)  # Store grid_image in the list
							else:
								break

					button_forward.config(command=lambda: update_grid(index + 9))
					button_back.config(command=lambda: update_grid(index - 9))

					if index == 0:
						button_back.config(state=DISABLED)
					else:
						button_back.config(state=NORMAL)

					if index + 9 >= len(cut_image_list):
						button_forward.config(state=DISABLED)
					else:
						button_forward.config(state=NORMAL)

					# Set button placement for 3x3 grid view
					button_back.place(x=330, y=10)
					button_forward.place(x=650, y=10)
					button_grid.place(x=470, y=10)


			def forward2(image_number):
				if grid_mode:
					update_grid(image_number)
				else:
					update_single(image_number)

			def back2(image_number):
				if grid_mode:
					update_grid(image_number)
				else:
					update_single(image_number)

			button_back = Button(newWindow2, text="<<", command=lambda: back2(0), state=DISABLED)
			button_forward = Button(newWindow2, text=">>", command=lambda: forward2(1))
			button_grid = Button(newWindow2, text="3x3 Grid", command=change_mode)

			button_back.place(x=200, y=10)
			button_forward.place(x=485, y=10)
			button_grid.place(x=330, y=10)

			update_single(0)

			return

		button_back = Button(newWindow, text="<<", command=back, state=DISABLED)
		button_exit = Button(newWindow, text="Exit Program", command=newWindow.destroy)
		button_forward = Button(newWindow, text=">>", command=lambda: forward(2))
		button_select = Button(newWindow, text="Select Cut", command=lambda: select_cut(image_index, init_image_list))
		button_h5 = Button(newWindow, text="Create H5 File", command=lambda: create_h5(image_index, init_image_list))

		button_back.place(x=350, y=40)
		button_exit.place(x=470, y=80)
		button_select.place(x=475, y=40)
		button_h5.place(x=470, y=950)
		button_forward.place(x=635, y=40)

		
		return
	


	def project_cut():
		from matplotlib.widgets import PolygonSelector
		from matplotlib.path import Path
		from datetime import datetime
		from tqdm import tqdm
		import pickle

		def create_heatmap_and_scatterplot_proj(x_data: np.ndarray, y_data: np.ndarray):

			# Create a 2D histogram using the x and y data
			num_bins = int(entry_bins_RvE.get())
			plt.rcParams['figure.figsize'] = [10, 10]

			heatmap, xedges, yedges = np.histogram2d(x_data, y_data, bins=num_bins, density=True)

			# Create a heatmap image using the 2D histogram
			extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
			plt.imshow(heatmap.T, extent=extent, origin='lower')

			# Add a callback function that is called when the user clicks and drags
			# on the heatmap to draw a polygon
			def onselect(verts):

				if verts:
					# Create a path from the selected vertices
					path = Path(verts)

					# Find all the points in the x and y data that lie inside the path
					inside = path.contains_points(np.stack([x_data, y_data], axis=1))

					# Create a scatter plot of the points that lie inside the path
					scatter_plot = plt.scatter(x_data[inside], y_data[inside], c="r", alpha=0.5)
					plt.imshow(heatmap.T, extent=extent, origin='lower')

					# dd/mm/YY H:M:S
					# datetime object containing current date and time
					now = datetime.now()

					dt_string = now.strftime("PROJECT_Date_%m_%d_%Y.png")
					image_string = now.strftime("PROJECT_Date_%m_%d_%Y")

					dt_string = str(rand_num) + dt_string
					image_string = str(rxand_num) + image_string

					full_path = os.path.join(sub_mymainpath, dt_string)

					global imageProject_path
					imageProject_path = os.path.join(sub_mymainpath, image_string)

					plt.savefig(full_path)
					os.remove(full_path)

					# Return the indices of the points that lie inside the path
					return np.where(inside)[0]

				# Return an empty list of indices if no polygon is selected
				return []

			# Create a PolygonSelector widget and connect it to the heatmap
			selector = PolygonSelector(plt.gca(), onselect, props=dict(color="r"))

			# Show the heatmap and any plotted data on it

			plt.show()

			return onselect(selector.verts)

		
		def create_energy_histogram(project_indices, num_bins):
			from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
			calib_point_1 = (0.806, 156745)
			calib_point_2 = (1.679, 320842)
			#calib_point_1 = (0.806, 157600)
			#calib_point_2 = (1.679, 275300)
			#calib_point_1 = (0.303, 84672)
			#calib_point_2 = (2.150, 374439)
			#calib_point_1 = (0.806, 157700)
			#calib_point_2 = (1.679, 308200)

			# Convert calibration points to MeV
			energy_1, channel_1 = calib_point_1
			energy_2, channel_2 = calib_point_2
			energy_scale_factor = (energy_2 - energy_1) / (channel_2 - channel_1)
			energy_offset = energy_1 - energy_scale_factor * channel_1

			# Convert tot_energy to MeV
			#x_data = np.array(tot_energy).T # use this for uncalibrated spectrum
			x_data = np.array(tot_energy).T * energy_scale_factor + energy_offset

			# Select the x_data values corresponding to the project_indices
			global projected_x_data
			projected_x_data = x_data[project_indices]

			# Create a new top-level window
			new_window = Toplevel()
			new_window.title("Projected Cut Energy Histogram")
			new_window.geometry("1000x1300")  # Set the width and height of the window

			# Create the energy histogram using the create_energy_histogram() function
			fig, ax = plt.subplots()
			num_bins = int(entry_bins_RvE.get())
			ax.hist(projected_x_data, bins=num_bins, alpha=0.75, edgecolor="k")
			ax.set_title("Projected Cut Energy Histogram")
			ax.set_xlabel("Integrated Charge")
			ax.set_ylabel("Counts")

			# Display the energy histogram in the Tkinter window
			canvas = FigureCanvasTkAgg(fig, master=new_window)
			canvas.draw()
			canvas.get_tk_widget().place(x=0, y=0)

			# Create a button with the label "Fit Data" and link it to the fit_data function
			fit_button = Button(new_window, text="Fit Data", command=plot_spectrum_multi_proj)
			fit_button.place(x=480, y=50) 
				

		# Allow for full fitting procedure if button clicked
		def plot_spectrum_multi_proj():
			plt.close()
			global peaks, peak_active, peak_data, peak_handle, peak_info
			# plot the histogram
			num_bins = int(entry_bins_RvE.get())
			fig, ax = plt.subplots()
			n, bins, patches = ax.hist(projected_x_data, bins=num_bins)
			y_hist, x_hist = np.histogram(projected_x_data, bins=num_bins)
			x_hist = (x_hist[:-1] + x_hist[1:]) / 2
			
			peak_handle, = plt.plot([], [], 'o', color='black', markersize=10, alpha=0.7)

			# keep track of the last left-click point
			last_left_click = None

			def onclick(event):
				global peak_active, peak_handle, peak_info, horizontal_line
				if event.button == 1:  # Left mouse button
					x, y = event.xdata, event.ydata
					plt.plot(x, y, 'ro', markersize=10)
					plt.axvline(x, color='r', linestyle='--')
					plt.draw()
					peak_active = x

				elif event.button == 3:  # Right mouse button
					if peak_active is not None:
						x, y = event.xdata, event.ydata
						plt.plot(x, y, 'go', markersize=10)
						plt.draw()

						idx = np.argmin(np.abs(x_hist - peak_active))
						mu = peak_active
						sigma = np.abs(x - peak_active)
						amp = y_hist[idx] * np.sqrt(2 * np.pi) * sigma
						peak_info.extend([amp, mu, sigma, 1])

						horizontal_line, = plt.plot([peak_active, x], [y, y], color='green', linestyle='--')

						peak_active = None
						plt.draw()

			# initialize peak detection variables
			peaks = []
			peak_data = []
			peak_active = None
			peak_info = []

			# connect the click event to the plot
			cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)

			title1 = "Left Click to Select Peak Amp and Mu"
			title2 = "\nRight Click to Select Peak Sigma"

			# Calculate the position for each part of the title
			x1, y1 = 0.5, 1.10
			x2, y2 = 0.5, 1.05

			# Set the title using ax.annotate() and the ax.transAxes transform
			ax.annotate(title1, (x1, y1), xycoords='axes fraction', fontsize=12, color='red', ha='center', va='center')
			ax.annotate(title2, (x2, y2), xycoords='axes fraction', fontsize=12, color='green', ha='center', va='center')

			# show the plot
			plt.show()

			# Send peak_info for fitting when the plot is closed
			#print('INITIAL GUESSES:\n',peak_info)

			# Print initial guesses
			print("INITIAL GUESSES:")
			for i in range(0, len(peak_info), 4):
				print(f"Peak {i//4 + 1}: Amp={peak_info[i]}, Mu={peak_info[i+1]}, Sigma={peak_info[i+2]}, Lambda={peak_info[i+3]}")


			fit_multi_peaks_proj(projected_x_data, peak_info, num_bins, x_hist, y_hist)
			plt.close()
			
			
		def fit_multi_peaks_proj(projected_x_data, peak_info, num_bins, x_hist, y_hist):
			from scipy.special import erfc
			from scipy.special import erfcx
			from scipy.optimize import curve_fit
			from scipy.stats import chisquare, chi2
			from matplotlib.offsetbox import AnchoredOffsetbox, TextArea


			def safe_exp(x, min_exp_arg=None, max_exp_arg=None):
				min_exp_arg = min_exp_arg if min_exp_arg is not None else -np.inf
				max_exp_arg = max_exp_arg if max_exp_arg is not None else np.finfo(np.float64).maxexp - 10
				return np.exp(np.clip(x, min_exp_arg, max_exp_arg))

			
			def emg_stable(x, amplitude, mu, sigma, lambda_):
				exp_arg = 0.5 * lambda_ * (2 * mu + lambda_ * sigma**2 - 2 * x)
				erfc_arg = (mu + lambda_ * sigma**2 - x) / (np.sqrt(2) * sigma)
				#print("lambda_: ", lambda_)
				#print("mu: ", mu)
				#print("sigma: ", sigma)
				#print("x: ", x)
				return 0.5 * amplitude * lambda_ * safe_exp(exp_arg - erfc_arg**2) * erfcx(erfc_arg)

			def composite_emg(x, *params):
				result = np.zeros_like(x)
				for i in range(0, len(params), 4):
					result += emg_stable(x, *params[i:i + 4])
				return result


			# Set the threshold for y_hist, adjust it based on your specific requirements
			y_hist_threshold = 1e5

			# Filter the data based on the threshold
			valid_indices = y_hist < y_hist_threshold
			filtered_x_hist = x_hist[valid_indices]
			filtered_y_hist = y_hist[valid_indices]


			# Fit the composite EMG function to the data
			popt, pcov = curve_fit(composite_emg, filtered_x_hist, filtered_y_hist, p0=peak_info, maxfev=1000000)
			fitted_emg = composite_emg(filtered_x_hist, *popt)
			#print('FINAL FIT PARAMETERS:', [*popt])
			# Print final fit parameters
			print("FINAL FIT PARAMETERS:")
			for i in range(0, len(popt), 4):
				print(f"Peak {i//4 + 1}: Amp={peak_info[i]}, Mu={peak_info[i+1]}, Sigma={peak_info[i+2]}, Lambda={peak_info[i+3]}")

			# Print final fit parameters
			def display_fit_parameters(peak_info, popt, fixed_list=None):
				fit_params_window = Toplevel()
				fit_params_window.title("Final Fit Parameters")
				fit_params_window.geometry("700x200")
				output_text = Text(fit_params_window, wrap=WORD)
				output_text.pack(expand=True, fill=BOTH)

				param_names = ['Amp', 'Mu', 'Sigma', 'Lambda']
				idx = 0
				fixed_param_idx = 0

				# If fixed_list is not provided, create a list of all False values
				if fixed_list is None:
					fixed_list = [False] * len(peak_info)

				for i in range(0, len(peak_info), 4):
					peak_label = f"Peak {(i // 4) + 1}: "
					for j in range(4):
						if fixed_list[i + j]:
							peak_label += f"*{param_names[j]}={fixed_params[fixed_param_idx]}, "
							fixed_param_idx += 1
						else:
							peak_label += f"{param_names[j]}={popt[idx]}, "
							idx += 1
					output_text.insert(END, peak_label + "\n")


			display_fit_parameters(peak_info, popt)

			# Calibration points
			calib_point_1 = (0.806, 156745)
			calib_point_2 = (1.679, 320842)
			#calib_point_1 = (0.806, 157600)
			#calib_point_2 = (1.679, 275300)
			#calib_point_1 = (0.303, 84672)
			#calib_point_2 = (2.150, 374439)
			#calib_point_1 = (0.806, 157700)
			#calib_point_2 = (1.679, 308200)

			# Convert calibration points to MeV
			energy_1, channel_1 = calib_point_1
			energy_2, channel_2 = calib_point_2
			energy_scale_factor = (energy_2 - energy_1) / (channel_2 - channel_1)
			energy_offset = energy_1 - energy_scale_factor * channel_1

			# Convert filtered_x_hist and tot_energy to MeV
			filtered_x_hist_MeV = filtered_x_hist * energy_scale_factor + energy_offset
			tot_energy_MeV = np.array(projected_x_data) * energy_scale_factor + energy_offset

			# Plot the histogram
			plt.bar(filtered_x_hist, filtered_y_hist, width=np.diff(filtered_x_hist)[0], alpha=0.6, label="Histogram")


			plt.rcParams['figure.figsize'] = [10, 10]
			plt.title(f'Energy Spectrum', fontdict = {'fontsize' : 20})
			plt.xlabel(f'Energy (MeV)', fontdict = {'fontsize' : 20})  # Update the x-axis label
			plt.ylabel(f'Counts',fontdict = {'fontsize' : 20})
			# Plot the fitted curve
			x = np.linspace(min(filtered_x_hist), max(filtered_x_hist), 1000)
			y = composite_emg(x, *popt)
			plt.plot(x, y, 'r--', label="Fit")
			plt.legend()

			plt.show()
			plt.close()

		
		tot_eng_array = np.array(tot_energy).T
		len_list_array = np.array(len_list).T
		a=-1
		b=10
		max_eng = np.max(tot_eng_array)
		min_eng = np.min(tot_eng_array)

		max_len = np.max(len_list_array)
		min_len = np.min(len_list_array)

		norm_tot_eng = (b-a)*(tot_eng_array - min_eng)/(max_eng - min_eng)+a
		norm_len_list = (b-a)*(len_list_array - min_len)/(max_len - min_len)+a
		

		global project_indices
		project_indices = create_heatmap_and_scatterplot_proj(norm_tot_eng, norm_len_list)
		plt.close()
		global rand_num
		rand_num = random.randrange(0,1000000,1)
		num_bins_proj = int(entry_bins_RvE.get())

		create_energy_histogram(project_indices, num_bins = num_bins_proj)


	def project_cut_y():
		from matplotlib.widgets import PolygonSelector
		from matplotlib.path import Path
		from datetime import datetime
		from tqdm import tqdm
		import pickle

		def create_heatmap_and_scatterplot_proj(x_data: np.ndarray, y_data: np.ndarray):

			# Create a 2D histogram using the x and y data
			num_bins = int(entry_bins_RvE.get())
			plt.rcParams['figure.figsize'] = [10, 10]

			heatmap, xedges, yedges = np.histogram2d(x_data, y_data, bins=num_bins, density=True)

			# Create a heatmap image using the 2D histogram
			extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
			plt.imshow(heatmap.T, extent=extent, origin='lower')

			# Add a callback function that is called when the user clicks and drags
			# on the heatmap to draw a polygon
			def onselect(verts):

				if verts:
					# Create a path from the selected vertices
					path = Path(verts)

					# Find all the points in the x and y data that lie inside the path
					inside = path.contains_points(np.stack([x_data, y_data], axis=1))

					# Create a scatter plot of the points that lie inside the path
					scatter_plot = plt.scatter(x_data[inside], y_data[inside], c="r", alpha=0.5)
					plt.imshow(heatmap.T, extent=extent, origin='lower')

					# dd/mm/YY H:M:S
					# datetime object containing current date and time
					now = datetime.now()

					dt_string = now.strftime("PROJECT_Date_%m_%d_%Y.png")
					image_string = now.strftime("PROJECT_Date_%m_%d_%Y")

					dt_string = str(rand_num) + dt_string
					image_string = str(rand_num) + image_string

					full_path = os.path.join(sub_mymainpath, dt_string)

					global imageProject_path
					imageProject_path = os.path.join(sub_mymainpath, image_string)

					plt.savefig(full_path)
					os.remove(full_path)

					# Return the indices of the points that lie inside the path
					return np.where(inside)[0]

				# Return an empty list of indices if no polygon is selected
				return []

			# Create a PolygonSelector widget and connect it to the heatmap
			selector = PolygonSelector(plt.gca(), onselect, props=dict(color="r"))

			# Show the heatmap and any plotted data on it

			plt.show()

			return onselect(selector.verts)

		
		def create_energy_histogram(project_indices, num_bins):
			from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
			y_data = np.array(len_list).T 


			# Select the x_data values corresponding to the project_indices
			global projected_y_data
			projected_y_data = y_data[project_indices]

			# Create a new top-level window
			new_window = Toplevel()
			new_window.title("Projected Cut Range Histogram")
			new_window.geometry("1000x1300")  # Set the width and height of the window

			# Create the energy histogram using the create_energy_histogram() function
			fig, ax = plt.subplots()
			num_bins = int(entry_bins_RvE.get())
			ax.hist(projected_y_data, bins=num_bins, alpha=0.75, edgecolor="k")
			ax.set_title("Projected Cut Range Histogram")
			ax.set_xlabel("Range (mm)")
			ax.set_ylabel("Counts")

			# Display the energy histogram in the Tkinter window
			canvas = FigureCanvasTkAgg(fig, master=new_window)
			canvas.draw()
			canvas.get_tk_widget().place(x=0, y=0)

			# Create a button with the label "Fit Data" and link it to the fit_data function
			fit_button = Button(new_window, text="Fit Data", command=plot_spectrum_multi_proj)
			fit_button.place(x=480, y=50) 
				

		# Allow for full fitting procedure if button clicked
		def plot_spectrum_multi_proj():
			plt.close()
			global peaks, peak_active, peak_data, peak_handle, peak_info
			# plot the histogram
			num_bins = int(entry_bins_RvE.get())
			fig, ax = plt.subplots()
			n, bins, patches = ax.hist(projected_y_data, bins=num_bins)
			y_hist, x_hist = np.histogram(projected_y_data, bins=num_bins)
			x_hist = (x_hist[:-1] + x_hist[1:]) / 2
			
			peak_handle, = plt.plot([], [], 'o', color='black', markersize=10, alpha=0.7)

			# keep track of the last left-click point
			last_left_click = None

			def onclick(event):
				global peak_active, peak_handle, peak_info, horizontal_line
				if event.button == 1:  # Left mouse button
					x, y = event.xdata, event.ydata
					plt.plot(x, y, 'ro', markersize=10)
					plt.axvline(x, color='r', linestyle='--')
					plt.draw()
					peak_active = x

				elif event.button == 3:  # Right mouse button
					if peak_active is not None:
						x, y = event.xdata, event.ydata
						plt.plot(x, y, 'go', markersize=10)
						plt.draw()

						idx = np.argmin(np.abs(x_hist - peak_active))
						mu = peak_active
						sigma = np.abs(x - peak_active)
						amp = y_hist[idx] * np.sqrt(2 * np.pi) * sigma
						peak_info.extend([amp, mu, sigma, 1])

						horizontal_line, = plt.plot([peak_active, x], [y, y], color='green', linestyle='--')

						peak_active = None
						plt.draw()

			# initialize peak detection variables
			peaks = []
			peak_data = []
			peak_active = None
			peak_info = []

			# connect the click event to the plot
			cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)

			title1 = "Left Click to Select Peak Amp and Mu"
			title2 = "\nRight Click to Select Peak Sigma"

			# Calculate the position for each part of the title
			x1, y1 = 0.5, 1.10
			x2, y2 = 0.5, 1.05

			# Set the title using ax.annotate() and the ax.transAxes transform
			ax.annotate(title1, (x1, y1), xycoords='axes fraction', fontsize=12, color='red', ha='center', va='center')
			ax.annotate(title2, (x2, y2), xycoords='axes fraction', fontsize=12, color='green', ha='center', va='center')

			# show the plot
			plt.show()

			# Send peak_info for fitting when the plot is closed
			#print('INITIAL GUESSES:\n',peak_info)

			# Print initial guesses
			print("INITIAL GUESSES:")
			for i in range(0, len(peak_info), 4):
				print(f"Peak {i//4 + 1}: Amp={peak_info[i]}, Mu={peak_info[i+1]}, Sigma={peak_info[i+2]}, Lambda={peak_info[i+3]}")


			fit_multi_peaks_proj(projected_y_data, peak_info, num_bins, x_hist, y_hist)
			plt.close()
			
			
		def fit_multi_peaks_proj(projected_y_data, peak_info, num_bins, x_hist, y_hist):
			from scipy.special import erfc
			from scipy.special import erfcx
			from scipy.optimize import curve_fit
			from scipy.stats import chisquare, chi2
			from matplotlib.offsetbox import AnchoredOffsetbox, TextArea


			def safe_exp(x, min_exp_arg=None, max_exp_arg=None):
				min_exp_arg = min_exp_arg if min_exp_arg is not None else -np.inf
				max_exp_arg = max_exp_arg if max_exp_arg is not None else np.finfo(np.float64).maxexp - 10
				return np.exp(np.clip(x, min_exp_arg, max_exp_arg))

			
			def emg_stable(x, amplitude, mu, sigma, lambda_):
				exp_arg = 0.5 * lambda_ * (2 * mu + lambda_ * sigma**2 - 2 * x)
				erfc_arg = (mu + lambda_ * sigma**2 - x) / (np.sqrt(2) * sigma)
				#print("lambda_: ", lambda_)
				#print("mu: ", mu)
				#print("sigma: ", sigma)
				#print("x: ", x)
				return 0.5 * amplitude * lambda_ * safe_exp(exp_arg - erfc_arg**2) * erfcx(erfc_arg)

			def composite_emg(x, *params):
				result = np.zeros_like(x)
				for i in range(0, len(params), 4):
					result += emg_stable(x, *params[i:i + 4])
				return result


			# Set the threshold for y_hist, adjust it based on your specific requirements
			y_hist_threshold = 1e5

			# Filter the data based on the threshold
			valid_indices = y_hist < y_hist_threshold
			filtered_x_hist = x_hist[valid_indices]
			filtered_y_hist = y_hist[valid_indices]


			# Fit the composite EMG function to the data
			popt, pcov = curve_fit(composite_emg, filtered_x_hist, filtered_y_hist, p0=peak_info, maxfev=1000000)
			fitted_emg = composite_emg(filtered_x_hist, *popt)
			#print('FINAL FIT PARAMETERS:', [*popt])
			# Print final fit parameters
			print("FINAL FIT PARAMETERS:")
			for i in range(0, len(popt), 4):
				print(f"Peak {i//4 + 1}: Amp={peak_info[i]}, Mu={peak_info[i+1]}, Sigma={peak_info[i+2]}, Lambda={peak_info[i+3]}")

			# Print final fit parameters
			def display_fit_parameters(peak_info, popt, fixed_list=None):
				fit_params_window = Toplevel()
				fit_params_window.title("Final Fit Parameters")
				fit_params_window.geometry("700x200")
				output_text = Text(fit_params_window, wrap=WORD)
				output_text.pack(expand=True, fill=BOTH)

				param_names = ['Amp', 'Mu', 'Sigma', 'Lambda']
				idx = 0
				fixed_param_idx = 0

				# If fixed_list is not provided, create a list of all False values
				if fixed_list is None:
					fixed_list = [False] * len(peak_info)

				for i in range(0, len(peak_info), 4):
					peak_label = f"Peak {(i // 4) + 1}: "
					for j in range(4):
						if fixed_list[i + j]:
							peak_label += f"*{param_names[j]}={fixed_params[fixed_param_idx]}, "
							fixed_param_idx += 1
						else:
							peak_label += f"{param_names[j]}={popt[idx]}, "
							idx += 1
					output_text.insert(END, peak_label + "\n")


			display_fit_parameters(peak_info, popt)


			# Plot the histogram
			plt.bar(filtered_x_hist, filtered_y_hist, width=np.diff(filtered_x_hist)[0], alpha=0.6, label="Histogram")

			plt.rcParams['figure.figsize'] = [10, 10]
			plt.title(f'Range Spectrum', fontdict = {'fontsize' : 20})
			plt.xlabel(f'Range (mm)', fontdict = {'fontsize' : 20})  # Update the x-axis label
			plt.ylabel(f'Counts',fontdict = {'fontsize' : 20})
			# Plot the fitted curve
			x = np.linspace(min(filtered_x_hist), max(filtered_x_hist), 1000)
			y = composite_emg(x, *popt)
			plt.plot(x, y, 'r--', label="Fit")
			plt.legend()
			
			plt.show()
			plt.close()

		
		tot_eng_array = np.array(tot_energy).T
		len_list_array = np.array(len_list).T
		a=-1
		b=10
		max_eng = np.max(tot_eng_array)
		min_eng = np.min(tot_eng_array)

		max_len = np.max(len_list_array)
		min_len = np.min(len_list_array)

		norm_tot_eng = (b-a)*(tot_eng_array - min_eng)/(max_eng - min_eng)+a
		norm_len_list = (b-a)*(len_list_array - min_len)/(max_len - min_len)+a
		

		global project_indices
		project_indices = create_heatmap_and_scatterplot_proj(norm_tot_eng, norm_len_list)
		plt.close()
		global rand_num
		rand_num = random.randrange(0,1000000,1)
		num_bins_proj = int(entry_bins_RvE.get())

		create_energy_histogram(project_indices, num_bins = num_bins_proj)



	def cut():
		from matplotlib.widgets import PolygonSelector
		from matplotlib.path import Path
		from datetime import datetime
		from tqdm import tqdm
		import pickle
		import torch
		from datetime import datetime
		

		def create_heatmap_and_scatterplot(x_data: np.ndarray, y_data: np.ndarray):

			# Create a 2D histogram using the x and y data
			num_bins = int(entry_bins_RvE.get())
			plt.rcParams['figure.figsize'] = [10, 10]
			
			heatmap, xedges, yedges = np.histogram2d(x_data, y_data, bins=num_bins, density=True)

			# Create a heatmap image using the 2D histogram
			extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
			plt.imshow(heatmap.T, extent=extent, origin='lower')

			# Add a callback function that is called when the user clicks and drags
			# on the heatmap to draw a polygon
			def onselect(verts):
					
				
				if verts:
					# Create a path from the selected vertices
					path = Path(verts)

					# Find all the points in the x and y data that lie inside the path
					inside = path.contains_points(np.stack([x_data, y_data], axis=1))
				
					# Create a scatter plot of the points that lie inside the path
					scatter_plot = plt.scatter(x_data[inside], y_data[inside], c="r", alpha=0.5)
					plt.imshow(heatmap.T, extent=extent, origin='lower')

					# dd/mm/YY H:M:S
					# datetime object containing current date and time
					now = datetime.now()

					dt_string = now.strftime("CUT_Date_%m_%d_%Y.jpg")
					image_string = now.strftime("CUT_Date_%m_%d_%Y")

					dt_string = str(rand_num) + dt_string 
					image_string = str(rand_num) + image_string

					full_path = os.path.join(sub_mymainpath, dt_string)

					global imageCut_path
					imageCut_path = os.path.join(sub_mymainpath, image_string)
					plt.savefig(full_path)

					# Return the indices of the points that lie inside the path
					return np.where(inside)[0]

				# Return an empty list of indices if no polygon is selected
				return []

			# Create a PolygonSelector widget and connect it to the heatmap
			selector = PolygonSelector(plt.gca(), onselect, props=dict(color="r"))

			# Show the heatmap and any plotted data on it
			
			plt.show()
			

			return onselect(selector.verts)


		tot_eng_array = np.array(tot_energy).T
		len_list_array = np.array(len_list).T
		a=-1
		b=10
		max_eng = np.max(tot_eng_array)
		min_eng = np.min(tot_eng_array)

		max_len = np.max(len_list_array)
		min_len = np.min(len_list_array)

		norm_tot_eng = (b-a)*(tot_eng_array - min_eng)/(max_eng - min_eng)+a
		norm_len_list = (b-a)*(len_list_array - min_len)/(max_len - min_len)+a
		

		global cut_indices
		cut_indices = create_heatmap_and_scatterplot(norm_tot_eng, norm_len_list)
		plt.close()
		global rand_num
		rand_num = random.randrange(0,1000000,1)

		# save images full_path

		#print('indicies', cut_indices)


		def save_cutImages(cut_indices, chunk_num):
			def make_grid():
				"""
				"Create Training Data.ipynb"eate grid matrix of MM outline and energy bar, see spreadsheet below
				https://docs.google.com/spreadsheets/d/1_bbg6svfEph_g_Z002rmzTLu8yjQzuj_p50wqs7mMrI/edit?usp=sharing
				"""
				row = np.array([63, 47, 39, 31, 27, 23, 19, 15, 15, 11, 11, 7, 7, 7, 7, 3, 3, 3, 3, 3, 3, 3, 7, 7, 
					    7, 7, 11, 11, 15, 15, 19, 23, 27, 31, 39, 47, 63]) 

				to_row = np.array([87, 103, 111, 119, 123, 127, 131, 135, 135, 139, 139, 143, 143, 143, 143, 147, 
					       147, 147, 147, 147, 147, 148, 143, 143, 143, 144, 139, 140, 135, 136, 132, 128,
					       124, 120, 112, 104, 88]) 

				col = np.array([100, 84, 76, 68, 64, 60, 56, 52, 52, 48, 48, 44, 44, 44, 44, 40, 40, 40, 40, 40, 
					    40, 40, 44, 44, 44, 44, 48, 48, 52, 52, 56, 60, 64, 68, 76, 84, 100])

				to_col = np.array([124, 140, 148, 156, 160, 164, 168, 172, 172, 176, 176, 180, 180, 180, 180, 184, 
					       184, 184, 184, 184, 184, 184, 180, 180, 180, 180, 176, 176, 172, 172, 168, 164, 
					       160, 156, 148, 140, 124]) 

				all_row = np.array([i for i in range(3, 148, 4)])
				all_col = np.array([i for i in range(40, 185, 4)]) 

				full_image_size_width = 224
				full_image_size_length = 151
				mm_grid = np.zeros((full_image_size_length, full_image_size_width, 3))    
				mm_grid.fill(255)                                                     

				for i in range(len(row)):
					# draw grid columns, 0 = black
					mm_grid[row[i]:to_row[i], all_col[i], :] = 0
			
				for i in range(len(col)):
					# draw grid rows
					mm_grid[all_row[i], col[i]:to_col[i], :] = 0

				# Ensure that matrix is of integers
				mm_grid = mm_grid.astype(int) 

				# Draw engergy bar box
				mm_grid = make_box(mm_grid)

				return mm_grid


			def blue_range(pad_plane, rows):

				start_row = 140
				low_color = 0
				high_color = 35

				for i in range(rows):
					pad_plane[start_row:start_row+5, 8:17, 0] = low_color
					pad_plane[start_row:start_row+5, 8:17, 1] = high_color
					start_row = start_row - 5 
					low_color = low_color + 35
					high_color = high_color + 35

				return pad_plane


			def yellow_range(pad_plane, rows):

				start_row = 105
				color = 220

				for i in range(rows):
					pad_plane[start_row:start_row+5, 8:17, 2] = color
					start_row = start_row - 5 
					color = color - 15

				return pad_plane


			def orange_range(pad_plane, rows):

				start_row = 70
				color = 210
				for i in range(rows):
					pad_plane[start_row:start_row+5, 8:17, 1] = color - 15
					pad_plane[start_row:start_row+5, 8:17, 2] = color
					start_row = start_row - 5 
					color = color - 15

				return pad_plane


			def red_range(pad_plane, rows):

				start_row = 35
				color = 250

				for i in range(rows):
					pad_plane[start_row:start_row+5, 8:17, 0] = color
					pad_plane[start_row:start_row+5, 8:17, 1] = 50
					pad_plane[start_row:start_row+5, 8:17, 2] = 50
					start_row = start_row - 5 
					color = color - 15

				return pad_plane


			def tot_energy_to_mev(tot_energy):
				"""Convert tot_energy to energy in MeV using the line of best fit."""
				# Calculate the gradient and y-intercept for the line of best fit
				x1, y1 = 156745, 0.806 # Calibration values
				x2, y2 = 320842, 1.679
				gradient = (y2 - y1) / (x2 - x1)
				y_intercept = y1 - gradient * x1
				return gradient * tot_energy + y_intercept


			def fill_energy_bar(pad_plane, tot_energy):
				"""
				Fills the energy bar where the amount of pixels fired and the color corresponds to the energy of the track
				Max pixel_range should be 28 (7 rows for each color), so need to adjust accordingly.
				"""
				# Calculate the energy in MeV
				energy_mev = tot_energy_to_mev(tot_energy)

				# Calculate the proportion of the energy bar that should be filled
				proportion_filled = energy_mev / 3

				# Calculate how many rows should be filled
				total_rows = math.floor(proportion_filled * 28)

				# Fill the energy bar one row at a time
				if total_rows > 0:
					pad_plane = blue_range(pad_plane, rows=min(total_rows, 7))
				if total_rows > 7:
					pad_plane = yellow_range(pad_plane, rows=min(total_rows-7, 7))
				if total_rows > 14:
					pad_plane = orange_range(pad_plane, rows=min(total_rows-14, 7))
				if total_rows > 21:
					pad_plane = red_range(pad_plane, rows=min(total_rows-21, 7))

				return pad_plane


			def pos_odd_even(event_value):
				"""
				Makes correction to positive points if they are odd or even
				"""
				if event_value % 2 == 0:
					event_value = event_value + 1
					return event_value

				else:
					return event_value


			def neg_odd_even(event_value):
				"""
				Makes correction to negative points if they are odd or even
				"""
				if event_value % 2 == 0:
					event_value = event_value - 1
					return event_value

				else:
					return event_value


			def pt_shift(xset, yset):
				"""
				Shifts all points to the center of nearest pad for pad mapping
				"""
				for j in range(len(xset)):

					if xset[j] > 0:
						xset[j] = math.floor(xset[j])
						pos_adj_valx = pos_odd_even(xset[j])
						xset[j] = pos_adj_valx

					elif xset[j] < 0:
						xset[j] = math.ceil(xset[j])
						neg_adj_valx = neg_odd_even(xset[j])
						xset[j] = neg_adj_valx

					if yset[j] > 0:
						yset[j] = math.floor(yset[j])
						pos_adj_valy = pos_odd_even(yset[j])
						yset[j] = pos_adj_valy

					elif yset[j] < 0:
						yset[j] = math.ceil(yset[j])
						neg_adj_valy = neg_odd_even(yset[j])
						yset[j] = neg_adj_valy

				return xset, yset


			def make_box(mm_grid):
				"""
				Draws the box for the energy bar
				"""
				box_row = np.array([4, 4])
				to_box_row = np.array([145, 146])
				for_box_col = np.array([7, 17])

				box_col = np.array([7, 7])
				to_box_col = np.array([17, 17])
				for_box_row = np.array([4, 145])

				# Draw vertical lines of energy bar box
				for i in range(len(box_row)):
					mm_grid[box_row[i]:to_box_row[i], for_box_col[i], :] = 0
					mm_grid[for_box_row[i], box_col[i]:to_box_col[i], :] = 0

				return mm_grid


			def fill_padplane(xset, yset, eset, tot_energy):
				"""
				Fills the 2D pad plane grid for image creation
				"""
				pad_plane = make_grid()

				xset = np.array(xset)
				yset = np.array(yset)
				eset = np.array(eset)

				# pad plane mapping
				x = (35 + xset) * 2 + 42    # col value
				y = 145 - (35 + yset) * 2   # row value

				# create a dictionary to store (x,y) as keys and e as values
				d = {}
				for i in range(len(x)):
					key = (x[i], y[i])
					if key in d:
						d[key] += eset[i]
					else:
						d[key] = eset[i]

				# convert the dictionary back to arrays
				x = np.zeros(len(d))
				y = np.zeros(len(d))
				eset = np.zeros(len(d))
				for i, key in enumerate(d):
					x[i] = key[0]
					y[i] = key[1]
					eset[i] = d[key]

				# Find max E value and normalize
				energy = eset
				max_energy = np.max(energy)
				norm_energy = energy / max_energy


				# Fill in pad plane   
				for k in range(len(x)):
				
					if y[k] < 9:
						y[k] = y[k] + 4

					if x[k] < 50:
						x[k] = x[k] + 4

					if x[k] > 174:
						x[k] = x[k] - 4

					if y[k] > 53:
						y[k] = y[k] - 4

					if x[k] > 134:
						x[k] = x[k] - 4

					if y[k] > 93:
						y[k] = y[k] - 4

					if y[k] > 133:
						y[k] = y[k] - 4	

					if x[k] < 90:
						x[k] = x[k] + 4


					pad_plane[int(y[k])-1:int(y[k])+2, int(x[k])-1:int(x[k])+2, 0] = norm_energy[k] * 205

					pad_plane[int(y[k])-1:int(y[k])+2, int(x[k])-1:int(x[k])+2, 1] = norm_energy[k] * 240
				
				pad_plane = fill_energy_bar(pad_plane, tot_energy)

				return pad_plane


			def trace_image(padplane_image, trace):
				"""
				Creates a 2D image from trace data
				"""
				# Save plot as jpeg (only want RGB channels, not an alpha channel)
				# Need to take monitor dpi into account to get correct pixel size
				# Plot should have a pixel size of 73x224

				my_dpi = 96
				fig, ax = plt.subplots(figsize=(224/my_dpi, 73/my_dpi))

				x = np.linspace(0, len(trace)-1, len(trace))
				
				ax.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
				ax.spines['top'].set_visible(False)
				ax.spines['right'].set_visible(False)
				ax.spines['bottom'].set_visible(False)
				ax.spines['left'].set_visible(False)
				ax.fill_between(x, trace, color='b', alpha=1)
				rand_num = random.randrange(0,1000000,1)
				temp_strg = f'/mnt/projects/e21072/OfflineAnalysis/analysis_scripts/energy_depo_{rand_num}.jpg'
				plt.savefig(temp_strg, dpi=my_dpi)
				plt.close()

				# Load png plot as a matrix so that it can be appended to pad plane plot
				img = plt.imread(temp_strg)
				os.remove(temp_strg)
				rows,cols,colors = img.shape # gives dimensions for RGB array
				img_size = rows*cols*colors
				img_1D_vector = img.reshape(img_size)

				# you can recover the orginal image with:
				trace_image = img_1D_vector.reshape(rows,cols,colors)

				# append pad plane image with trace image
				complete_image = np.append(padplane_image, trace_image, axis=0)

				return complete_image

			def plot_track(cut_indices):
				import pickle
				import torch
				all_image_matricies = []
				pbar = tqdm(total=len(cut_indices))
				xHit_list = np.load(os.path.join(sub_mymainpath, 'xHit_list.npy'), allow_pickle=True)
				yHit_list = np.load(os.path.join(sub_mymainpath, 'yHit_list.npy'), allow_pickle=True)
				eHit_list = np.load(os.path.join(sub_mymainpath, 'eHit_list.npy'), allow_pickle=True)

				for event_num in cut_indices:
					xHit = xHit_list[event_num]
					yHit = yHit_list[event_num]
					eHit = eHit_list[event_num]

					trace = trace_list[event_num]

					energy = tot_energy[event_num]

					mm_grid = make_grid()
					pad_plane = np.repeat(mm_grid[np.newaxis, :, :], 1, axis=0)
					new_pad_plane = np.repeat(mm_grid[np.newaxis, :, :], 1, axis=0)

					    
					# Call pt_shift function to move all 2D pts to pad centers
					dset_0_copyx, dset_0_copyy = pt_shift(xHit, yHit)
						
					# Call fill_padplane to create 2D pad plane image
					pad_plane = np.append(pad_plane, new_pad_plane, axis=0)
					pad_plane[0] = fill_padplane(dset_0_copyx, dset_0_copyy, eHit, energy)

					# Call trace_image() to append trace to pad plane image
					complete_image = (trace_image(pad_plane[0], trace))
					all_image_matricies.append(complete_image)

					title = "Particle Track"
					plt.rcParams['figure.figsize'] = [7, 7]
					plt.title(f' Image {good_events[event_num]} of {title} Event:', fontdict = {'fontsize' : 20})
					plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
					str_event_num = f"run{run_num}_image_{event_num}.jpg"
					str_imgSave = os.path.join(imageCut_path, str_event_num)
					plt.imshow(complete_image)
					plt.savefig(str_imgSave)
					plt.close()
	
					pbar.update(n=1)
				
				

				del xHit_list 
				del yHit_list 
				del eHit_list

				return

			plot_track(cut_indices)
			return
	
		# Make directory for images in cut region
		print('NEW DIRECTORY', imageCut_path)
		os.makedirs(imageCut_path)
		
		# Process images in chunks to avoiding overloading memory
		chunk_size = 500
		num_images = len(cut_indices)
		print("Total Number of Image:", num_images)
		num_chunks = (num_images + chunk_size - 1) // chunk_size
		print("Total Number of Chunks:", num_chunks)
		chunk_num = 1

		pbar = tqdm(total=num_chunks)
		for chunk_idx in range(num_chunks):
			print(f"Processing Chunk {chunk_idx+1} of {num_chunks}")
			start_idx = chunk_idx * chunk_size
			end_idx = min((chunk_idx + 1) * chunk_size, num_images)

			chunk_indices = cut_indices[start_idx:end_idx]
			save_cutImages(chunk_indices, chunk_num)
			chunk_num += 1
			pbar.update(n=1)

			# Update the GUI and process pending events
			root.update_idletasks()
			root.update()

		print("All images have been processed")
		
		# Pickle cut_indices
		cut_indices_H5list = good_events[cut_indices]
		cut_indices_str = f"cut_indices_H5list.pkl"
		cut_indices_path = os.path.join(imageCut_path, cut_indices_str)
		with open(cut_indices_path, "wb") as file:
    			pickle.dump(cut_indices_H5list, file)

	def cut_sd(sd_size):
		from matplotlib.path import Path
		from datetime import datetime
		from tqdm import tqdm
		import pickle
		import torch
		from datetime import datetime

		print('SD SIZE', sd_size)

		if sd_size == 1:
			x_trapezoid_calibrated = [1.49, 1.49, 1.96, 1.96]
			y_trapezoid = [24.29, 41.39, 60.19, 31.79]
		else: 
			
			x_trapezoid_calibrated = [1.27, 1.27, 2.21, 2.21]
			y_trapezoid = [11.46, 45.97, 83.57, 26.46]

		
		def create_heatmap_and_scatterplot(x_data: np.ndarray, y_data: np.ndarray, x_trapezoid_calibrated, y_trapezoid):
			# Create a 2D histogram using the x and y data
			num_bins = int(entry_bins_RvE.get())
			plt.rcParams['figure.figsize'] = [10, 10]

			heatmap, xedges, yedges = np.histogram2d(x_data, y_data, bins=num_bins, density=True)

			# Create a heatmap image using the 2D histogram
			extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
			plt.imshow(heatmap.T, extent=extent, origin='lower')

			# Define the trapezoid vertices
			calib_point_1 = (0.806, 156745)
			calib_point_2 = (1.679, 320842)
			#calib_point_1 = (0.806, 157600)
			#calib_point_2 = (1.679, 275300)
			#calib_point_1 = (0.303, 84672)
			#calib_point_2 = (2.150, 374439)
			#calib_point_1 = (0.806, 157700)
			#calib_point_2 = (1.679, 308200)

			# Convert calibration points to Integrated Charge
			energy_1, channel_1 = calib_point_1
			energy_2, channel_2 = calib_point_2
			energy_scale_factor = (energy_2 - energy_1) / (channel_2 - channel_1)
			energy_offset = energy_1 - energy_scale_factor * channel_1

			# Define the trapezoid vertices
			x_trapezoid = (np.array(x_trapezoid_calibrated) - energy_offset) / energy_scale_factor

			# Scale the trapezoid vertices
			a=-1
			b=10
			max_eng = np.max(tot_eng_array)
			min_eng = np.min(tot_eng_array)

			max_len = np.max(len_list_array)
			min_len = np.min(len_list_array)
			x_trapezoid_scaled = (b - a) * (np.array(x_trapezoid) - min_eng) / (max_eng - min_eng) + a
			y_trapezoid_scaled = (b - a) * (np.array(y_trapezoid) - min_len) / (max_len - min_len) + a

			# Create a path from the scaled trapezoid vertices
			path = Path(list(zip(x_trapezoid_scaled, y_trapezoid_scaled)))

			# Find all the points in the x and y data that lie inside the path
			inside = path.contains_points(np.stack([x_data, y_data], axis=1))

			# Create a scatter plot of the points that lie inside the path
			scatter_plot = plt.scatter(x_data[inside], y_data[inside], c="r", alpha=0.5)
			plt.imshow(heatmap.T, extent=extent, origin='lower')

			# Plot the trapezoid on the heatmap
			plt.plot(*zip(*path.vertices, path.vertices[0]), color="r")

			# Save the plot with the cut applied
			now = datetime.now()
			dt_string = now.strftime("CUT_Date_%m_%d_%Y.jpg")
			image_string = now.strftime("CUT_Date_%m_%d_%Y")

			dt_string = str(rand_num) + dt_string
			image_string = str(rand_num) + image_string

			full_path = os.path.join(sub_mymainpath, dt_string)

			global imageCut_path
			imageCut_path = os.path.join(sub_mymainpath, image_string)

			plt.savefig(full_path)
			plt.show()
			plt.close()


			# Return the indices of the points that lie inside the path
			return np.where(inside)[0]


		tot_eng_array = np.array(tot_energy).T
		len_list_array = np.array(len_list).T
		a=-1
		b=10
		max_eng = np.max(tot_eng_array)
		min_eng = np.min(tot_eng_array)

		max_len = np.max(len_list_array)
		min_len = np.min(len_list_array)

		norm_tot_eng = (b-a)*(tot_eng_array - min_eng)/(max_eng - min_eng)+a
		norm_len_list = (b-a)*(len_list_array - min_len)/(max_len - min_len)+a
		

		global cut_indices
		cut_indices = create_heatmap_and_scatterplot(norm_tot_eng, norm_len_list, x_trapezoid_calibrated, y_trapezoid)
		plt.close()
		global rand_num
		rand_num = random.randrange(0,1000000,1)

		# save images full_path

		#print('indicies', cut_indices)

		def save_cutImages(cut_indices, chunk_num):
			def make_grid():
				"""
				"Create Training Data.ipynb"eate grid matrix of MM outline and energy bar, see spreadsheet below
				https://docs.google.com/spreadsheets/d/1_bbg6svfEph_g_Z002rmzTLu8yjQzuj_p50wqs7mMrI/edit?usp=sharing
				"""
				row = np.array([63, 47, 39, 31, 27, 23, 19, 15, 15, 11, 11, 7, 7, 7, 7, 3, 3, 3, 3, 3, 3, 3, 7, 7, 
					    7, 7, 11, 11, 15, 15, 19, 23, 27, 31, 39, 47, 63]) 

				to_row = np.array([87, 103, 111, 119, 123, 127, 131, 135, 135, 139, 139, 143, 143, 143, 143, 147, 
					       147, 147, 147, 147, 147, 148, 143, 143, 143, 144, 139, 140, 135, 136, 132, 128,
					       124, 120, 112, 104, 88]) 

				col = np.array([100, 84, 76, 68, 64, 60, 56, 52, 52, 48, 48, 44, 44, 44, 44, 40, 40, 40, 40, 40, 
					    40, 40, 44, 44, 44, 44, 48, 48, 52, 52, 56, 60, 64, 68, 76, 84, 100])

				to_col = np.array([124, 140, 148, 156, 160, 164, 168, 172, 172, 176, 176, 180, 180, 180, 180, 184, 
					       184, 184, 184, 184, 184, 184, 180, 180, 180, 180, 176, 176, 172, 172, 168, 164, 
					       160, 156, 148, 140, 124]) 

				all_row = np.array([i for i in range(3, 148, 4)])
				all_col = np.array([i for i in range(40, 185, 4)]) 

				full_image_size_width = 224
				full_image_size_length = 151
				mm_grid = np.zeros((full_image_size_length, full_image_size_width, 3))    
				mm_grid.fill(255)                                                     

				for i in range(len(row)):
					# draw grid columns, 0 = black
					mm_grid[row[i]:to_row[i], all_col[i], :] = 0
			
				for i in range(len(col)):
					# draw grid rows
					mm_grid[all_row[i], col[i]:to_col[i], :] = 0

				# Ensure that matrix is of integers
				mm_grid = mm_grid.astype(int) 

				# Draw engergy bar box
				mm_grid = make_box(mm_grid)

				return mm_grid


			def blue_range(pad_plane, rows):

				start_row = 140
				low_color = 0
				high_color = 35

				for i in range(rows):
					pad_plane[start_row:start_row+5, 8:17, 0] = low_color
					pad_plane[start_row:start_row+5, 8:17, 1] = high_color
					start_row = start_row - 5 
					low_color = low_color + 35
					high_color = high_color + 35

				return pad_plane


			def yellow_range(pad_plane, rows):

				start_row = 105
				color = 220

				for i in range(rows):
					pad_plane[start_row:start_row+5, 8:17, 2] = color
					start_row = start_row - 5 
					color = color - 15

				return pad_plane


			def orange_range(pad_plane, rows):

				start_row = 70
				color = 210
				for i in range(rows):
					pad_plane[start_row:start_row+5, 8:17, 1] = color - 15
					pad_plane[start_row:start_row+5, 8:17, 2] = color
					start_row = start_row - 5 
					color = color - 15

				return pad_plane


			def red_range(pad_plane, rows):

				start_row = 35
				color = 250

				for i in range(rows):
					pad_plane[start_row:start_row+5, 8:17, 0] = color
					pad_plane[start_row:start_row+5, 8:17, 1] = 50
					pad_plane[start_row:start_row+5, 8:17, 2] = 50
					start_row = start_row - 5 
					color = color - 15

				return pad_plane


			def tot_energy_to_mev(tot_energy):
				"""Convert tot_energy to energy in MeV using the line of best fit."""
				# Calculate the gradient and y-intercept for the line of best fit
				x1, y1 = 156745, 0.806 # Calibration values
				x2, y2 = 320842, 1.679
				gradient = (y2 - y1) / (x2 - x1)
				y_intercept = y1 - gradient * x1
				return gradient * tot_energy + y_intercept


			def fill_energy_bar(pad_plane, tot_energy):
				"""
				Fills the energy bar where the amount of pixels fired and the color corresponds to the energy of the track
				Max pixel_range should be 28 (7 rows for each color), so need to adjust accordingly.
				"""
				# Calculate the energy in MeV
				energy_mev = tot_energy_to_mev(tot_energy)

				# Calculate the proportion of the energy bar that should be filled
				proportion_filled = energy_mev / 3

				# Calculate how many rows should be filled
				total_rows = math.floor(proportion_filled * 28)

				# Fill the energy bar one row at a time
				if total_rows > 0:
					pad_plane = blue_range(pad_plane, rows=min(total_rows, 7))
				if total_rows > 7:
					pad_plane = yellow_range(pad_plane, rows=min(total_rows-7, 7))
				if total_rows > 14:
					pad_plane = orange_range(pad_plane, rows=min(total_rows-14, 7))
				if total_rows > 21:
					pad_plane = red_range(pad_plane, rows=min(total_rows-21, 7))

				return pad_plane


			def pos_odd_even(event_value):
				"""
				Makes correction to positive points if they are odd or even
				"""
				if event_value % 2 == 0:
					event_value = event_value + 1
					return event_value

				else:
					return event_value


			def neg_odd_even(event_value):
				"""
				Makes correction to negative points if they are odd or even
				"""
				if event_value % 2 == 0:
					event_value = event_value - 1
					return event_value

				else:
					return event_value


			def pt_shift(xset, yset):
				"""
				Shifts all points to the center of nearest pad for pad mapping
				"""
				for j in range(len(xset)):

					if xset[j] > 0:
						xset[j] = math.floor(xset[j])
						pos_adj_valx = pos_odd_even(xset[j])
						xset[j] = pos_adj_valx

					elif xset[j] < 0:
						xset[j] = math.ceil(xset[j])
						neg_adj_valx = neg_odd_even(xset[j])
						xset[j] = neg_adj_valx

					if yset[j] > 0:
						yset[j] = math.floor(yset[j])
						pos_adj_valy = pos_odd_even(yset[j])
						yset[j] = pos_adj_valy

					elif yset[j] < 0:
						yset[j] = math.ceil(yset[j])
						neg_adj_valy = neg_odd_even(yset[j])
						yset[j] = neg_adj_valy

				return xset, yset


			def make_box(mm_grid):
				"""
				Draws the box for the energy bar
				"""
				box_row = np.array([4, 4])
				to_box_row = np.array([145, 146])
				for_box_col = np.array([7, 17])

				box_col = np.array([7, 7])
				to_box_col = np.array([17, 17])
				for_box_row = np.array([4, 145])

				# Draw vertical lines of energy bar box
				for i in range(len(box_row)):
					mm_grid[box_row[i]:to_box_row[i], for_box_col[i], :] = 0
					mm_grid[for_box_row[i], box_col[i]:to_box_col[i], :] = 0

				return mm_grid


			def fill_padplane(xset, yset, eset, tot_energy):
				"""
				Fills the 2D pad plane grid for image creation
				"""
				pad_plane = make_grid()

				xset = np.array(xset)
				yset = np.array(yset)
				eset = np.array(eset)

				# pad plane mapping
				x = (35 + xset) * 2 + 42    # col value
				y = 145 - (35 + yset) * 2   # row value

				# create a dictionary to store (x,y) as keys and e as values
				d = {}
				for i in range(len(x)):
					key = (x[i], y[i])
					if key in d:
						d[key] += eset[i]
					else:
						d[key] = eset[i]

				# convert the dictionary back to arrays
				x = np.zeros(len(d))
				y = np.zeros(len(d))
				eset = np.zeros(len(d))
				for i, key in enumerate(d):
					x[i] = key[0]
					y[i] = key[1]
					eset[i] = d[key]

				# Find max E value and normalize
				energy = eset
				max_energy = np.max(energy)
				norm_energy = energy / max_energy


				# Fill in pad plane   
				for k in range(len(x)):
				
					if y[k] < 9:
						y[k] = y[k] + 4

					if x[k] < 50:
						x[k] = x[k] + 4

					if x[k] > 174:
						x[k] = x[k] - 4

					if y[k] > 53:
						y[k] = y[k] - 4

					if x[k] > 134:
						x[k] = x[k] - 4

					if y[k] > 93:
						y[k] = y[k] - 4

					if y[k] > 133:
						y[k] = y[k] - 4	

					if x[k] < 90:
						x[k] = x[k] + 4


					pad_plane[int(y[k])-1:int(y[k])+2, int(x[k])-1:int(x[k])+2, 0] = norm_energy[k] * 205

					pad_plane[int(y[k])-1:int(y[k])+2, int(x[k])-1:int(x[k])+2, 1] = norm_energy[k] * 240
				
				pad_plane = fill_energy_bar(pad_plane, tot_energy)

				return pad_plane

			def trace_image(padplane_image, trace):
				"""
				Creates a 2D image from trace data
				"""
				# Save plot as jpeg (only want RGB channels, not an alpha channel)
				# Need to take monitor dpi into account to get correct pixel size
				# Plot should have a pixel size of 73x224

				my_dpi = 96
				fig, ax = plt.subplots(figsize=(224/my_dpi, 73/my_dpi))

				x = np.linspace(0, len(trace)-1, len(trace))
				
				ax.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
				ax.spines['top'].set_visible(False)
				ax.spines['right'].set_visible(False)
				ax.spines['bottom'].set_visible(False)
				ax.spines['left'].set_visible(False)
				ax.fill_between(x, trace, color='b', alpha=1)
				rand_num = random.randrange(0,1000000,1)
				temp_strg = f'/mnt/projects/e21072/OfflineAnalysis/analysis_scripts/energy_depo_{rand_num}.jpg'
				plt.savefig(temp_strg, dpi=my_dpi)
				plt.close()

				# Load png plot as a matrix so that it can be appended to pad plane plot
				img = plt.imread(temp_strg)
				os.remove(temp_strg)
				rows,cols,colors = img.shape # gives dimensions for RGB array
				img_size = rows*cols*colors
				img_1D_vector = img.reshape(img_size)

				# you can recover the orginal image with:
				trace_image = img_1D_vector.reshape(rows,cols,colors)

				# append pad plane image with trace image
				complete_image = np.append(padplane_image, trace_image, axis=0)

				return complete_image
			
			def plot_track(cut_indices):
				import pickle
				import torch
				all_image_matricies = []
				pbar = tqdm(total=len(cut_indices))
				xHit_list = np.load(os.path.join(sub_mymainpath, 'xHit_list.npy'), allow_pickle=True)
				yHit_list = np.load(os.path.join(sub_mymainpath, 'yHit_list.npy'), allow_pickle=True)
				eHit_list = np.load(os.path.join(sub_mymainpath, 'eHit_list.npy'), allow_pickle=True)

				for event_num in cut_indices:
					xHit = xHit_list[event_num]
					yHit = yHit_list[event_num]
					eHit = eHit_list[event_num]

					trace = trace_list[event_num]

					energy = tot_energy[event_num]

					mm_grid = make_grid()
					pad_plane = np.repeat(mm_grid[np.newaxis, :, :], 1, axis=0)
					new_pad_plane = np.repeat(mm_grid[np.newaxis, :, :], 1, axis=0)

					    
					# Call pt_shift function to move all 2D pts to pad centers
					dset_0_copyx, dset_0_copyy = pt_shift(xHit, yHit)
						
					# Call fill_padplane to create 2D pad plane image
					pad_plane = np.append(pad_plane, new_pad_plane, axis=0)
					pad_plane[0] = fill_padplane(dset_0_copyx, dset_0_copyy, eHit, energy)

					# Call trace_image() to append trace to pad plane image
					complete_image = (trace_image(pad_plane[0], trace))
					all_image_matricies.append(complete_image)

					title = "Particle Track"
					plt.rcParams['figure.figsize'] = [7, 7]
					plt.title(f' Image {good_events[event_num]} of {title} Event:', fontdict = {'fontsize' : 20})
					plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
					str_event_num = f"run{run_num}_image_{event_num}.jpg"
					str_imgSave = os.path.join(imageCut_path, str_event_num)
					plt.imshow(complete_image)
					plt.savefig(str_imgSave)
					plt.close()
	
					pbar.update(n=1)
				

				del xHit_list 
				del yHit_list 
				del eHit_list

				return

			plot_track(cut_indices)
			return

		# Make directory for images in cut region
		print('NEW DIRECTORY', imageCut_path)
		os.makedirs(imageCut_path)
		
		# Process images in chunks to avoiding overloading memory
		chunk_size = 500
		num_images = len(cut_indices)
		print("Total Number of Image:", num_images)
		num_chunks = (num_images + chunk_size - 1) // chunk_size
		print("Total Number of Chunks:", num_chunks)
		chunk_num = 1

		pbar = tqdm(total=num_chunks)
		for chunk_idx in range(num_chunks):
			print(f"Processing Chunk {chunk_idx+1} of {num_chunks}")
			start_idx = chunk_idx * chunk_size
			end_idx = min((chunk_idx + 1) * chunk_size, num_images)

			chunk_indices = cut_indices[start_idx:end_idx]
			save_cutImages(chunk_indices, chunk_num)
			chunk_num += 1
			pbar.update(n=1)

			# Update the GUI and process pending events
			root.update_idletasks()
			root.update()

		print("All images have been processed")
		
		# Pickle cut_indices
		cut_indices_H5list = good_events[cut_indices]
		cut_indices_str = f"cut_indices_H5list.pkl"
		cut_indices_path = os.path.join(imageCut_path, cut_indices_str)
		with open(cut_indices_path, "wb") as file:
    			pickle.dump(cut_indices_H5list, file)


	def sel_cut_sd_1():
		cut_sd(sd_size=1)

	def sel_cut_sd_2():
		cut_sd(sd_size=2)

	def on_focus_in(event):
		if event.widget.get() == event.widget.default_text:
			event.widget.delete(0, END)

	def on_focus_out(event):
		if event.widget.get() == '':
			event.widget.insert(0, event.widget.default_text)

	def close_out():
		entry_bins_RvE.destroy()
		button_spec.destroy()
		button_RvE_temp.destroy()
		button_cut.destroy()
		button_prev_cut.destroy()
		button_spec_point.destroy()
		entry_event_num.destroy()
		button_3D.destroy()
		button_eng_spec.destroy()
		button_track_angle.destroy()
		button_cnn.destroy()
		button_track_trace.destroy()
		button_proj_cut.destroy()
		button_cut_sd1.destroy()
		button_cut_sd2.destroy()
		button_proj_cut_y.destroy()


	entry_bins_RvE = Entry(root, borderwidth=5, width=12)
	entry_bins_RvE.default_text = 'Enter # of Bins'
	entry_bins_RvE.insert(0, entry_bins_RvE.default_text)
	entry_bins_RvE.bind('<FocusIn>', on_focus_in)
	entry_bins_RvE.bind('<FocusOut>', on_focus_out)
	canvas1.create_window(480, 115, window=entry_bins_RvE)

	button_spec = Button(root, text="Plot Range vs Energy", command=plot_spectrum)
	canvas1.create_window(480, 185, window=button_spec)

	entry_event_num = Entry(root, width=7)
	entry_event_num.default_text = "Event #"
	entry_event_num.insert(0, entry_event_num.default_text)
	entry_event_num.bind("<FocusIn>", on_focus_in)
	entry_event_num.bind("<FocusOut>", on_focus_out)
	canvas1.create_window(480, 245, window=entry_event_num)

	button_spec_point = Button(root, text="Plot RvE with Point", command=plot_spectrum_point)
	canvas1.create_window(480, 275, window=button_spec_point)

	button_cut = Button(root, text="Polygon Cut", command=cut)
	canvas1.create_window(387, 345, window=button_cut)

	button_cut_sd1 = Button(root, text="1 Std Dev Cut", command=sel_cut_sd_1)
	canvas1.create_window(480, 345, window=button_cut_sd1)

	button_cut_sd2 = Button(root, text="2 Std Dev Cut", command=sel_cut_sd_2)
	canvas1.create_window(578, 345, window=button_cut_sd2)

	button_prev_cut = Button(root, text="Previous Cuts", command=prev_cut)
	canvas1.create_window(480, 415, window=button_prev_cut)

	button_proj_cut = Button(root, text="Project Cut to X-axis", command=project_cut)
	canvas1.create_window(415, 485, window=button_proj_cut)

	button_proj_cut_y = Button(root, text="Project Cut to Y-axis", command=project_cut_y)
	canvas1.create_window(545, 485, window=button_proj_cut_y)
	
	# RvE Button
	button_RvE_temp = Button(text='Range vs Energy',fg='red',command=close_out)
	canvas1.create_window(163, 285, window=button_RvE_temp)

	# Energy Spectrum Button    
	button_eng_spec = Button(text='Energy Spectrum',fg='green', command=open_eng_spec)
	canvas1.create_window(163, 235, window=button_eng_spec)
	button_eng_spec["state"] = "disabled"

	# 3D Plot Button
	button_3D = Button(text='3D Event Plot',fg='green', command=open_3d_plot)
	canvas1.create_window(163, 335, window=button_3D)
	button_3D["state"] = "disabled"

	# Track with Trace Button
	button_track_trace = Button(text='Track with Trace',fg='green', command=open_track_trace)
	canvas1.create_window(163, 385, window=button_track_trace)
	button_track_trace["state"] = "disabled"

	# Track Angle Button
	button_track_angle = Button(text='Track Angles',fg='green', command=open_track_angles)
	canvas1.create_window(163, 435, window=button_track_angle)
	button_track_angle["state"] = "disabled"

	# ConvNet Button
	button_cnn = Button(text='ConvNet Track ID',fg='green', command=cnn)
	canvas1.create_window(163, 485, window=button_cnn)
	button_cnn["state"] = "disabled"


############################################################### Energy Spectrum
###############################################################

def energy_spectrum():

	import matplotlib.pyplot as plt
	import numpy as np
	from scipy.optimize import curve_fit
	from scipy.stats import chi2
	from scipy.optimize import minimize
	
	def plot_spectrum():
		# (energy, int_charge)
		calib_point_1 = (0.806, 156745)
		calib_point_2 = (1.679, 320842)
		#calib_point_1 = (0.806, 157600)
		#calib_point_2 = (1.679, 275300)
		#calib_point_1 = (0.303, 84672)
		#calib_point_2 = (2.150, 374439)
		#calib_point_1 = (0.806, 157700)
		#calib_point_2 = (1.679, 308200)

		# Convert calibration points to MeV
		energy_1, channel_1 = calib_point_1
		energy_2, channel_2 = calib_point_2
		energy_scale_factor = (energy_2 - energy_1) / (channel_2 - channel_1)
		energy_offset = energy_1 - energy_scale_factor * channel_1

		# Convert tot_energy to MeV
		tot_energy_MeV = np.array(tot_energy) * energy_scale_factor + energy_offset

		num_bins = int(entry_binsRvE.get())
		plt.hist(tot_energy_MeV, bins = num_bins)
		plt.rcParams['figure.figsize'] = [10, 10]
		plt.title(f'Energy Spectrum', fontdict = {'fontsize' : 20})
		plt.xlabel(f'Energy (MeV)', fontdict = {'fontsize' : 20})
		plt.ylabel(f'Counts',fontdict = {'fontsize' : 20})
		plt.show()

	def plot_spectrum_multi():
		global peaks, peak_active, peak_data, peak_handle, peak_info
		# plot the histogram
		num_bins = int(entry_binsRvE.get())
		fig, ax = plt.subplots()
		n, bins, patches = ax.hist(tot_energy, bins=num_bins)
		y_hist, x_hist = np.histogram(tot_energy, bins=num_bins)
		x_hist = (x_hist[:-1] + x_hist[1:]) / 2
		
		peak_handle, = plt.plot([], [], 'o', color='black', markersize=10, alpha=0.7)

		# keep track of the last left-click point
		last_left_click = None

		def onclick(event):
			global peak_active, peak_handle, peak_info, horizontal_line
			if event.button == 1:  # Left mouse button
				x, y = event.xdata, event.ydata
				plt.plot(x, y, 'ro', markersize=10)
				plt.axvline(x, color='r', linestyle='--')
				plt.draw()
				peak_active = x

			elif event.button == 3:  # Right mouse button
				if peak_active is not None:
					x, y = event.xdata, event.ydata
					plt.plot(x, y, 'go', markersize=10)
					plt.draw()

					idx = np.argmin(np.abs(x_hist - peak_active))
					mu = peak_active
					sigma = np.abs(x - peak_active)
					amp = y_hist[idx] * np.sqrt(2 * np.pi) * sigma
					peak_info.extend([amp, mu, sigma, 1])

					horizontal_line, = plt.plot([peak_active, x], [y, y], color='green', linestyle='--')

					peak_active = None
					plt.draw()

		# initialize peak detection variables
		peaks = []
		peak_data = []
		peak_active = None
		peak_info = []

		# connect the click event to the plot
		cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)

		title1 = "Left Click to Select Peak Amp and Mu"
		title2 = "\nRight Click to Select Peak Sigma"

		# Calculate the position for each part of the title
		x1, y1 = 0.5, 1.10
		x2, y2 = 0.5, 1.05

		# Set the title using ax.annotate() and the ax.transAxes transform
		ax.annotate(title1, (x1, y1), xycoords='axes fraction', fontsize=12, color='red', ha='center', va='center')
		ax.annotate(title2, (x2, y2), xycoords='axes fraction', fontsize=12, color='green', ha='center', va='center')

		# show the plot
		plt.show()

		# Send peak_info for fitting when the plot is closed
		#print('INITIAL GUESSES:\n',peak_info)

		# Print initial guesses
		print("INITIAL GUESSES:")
		for i in range(0, len(peak_info), 4):
			print(f"Peak {i//4 + 1}: Amp={peak_info[i]}, Mu={peak_info[i+1]}, Sigma={peak_info[i+2]}, Lambda={peak_info[i+3]}")


		fit_multi_peaks(peak_info, num_bins, x_hist, y_hist)
		
		
	def fit_multi_peaks(peak_info, num_bins, x_hist, y_hist):
		from scipy.special import erfc
		from scipy.special import erfcx
		from scipy.optimize import curve_fit
		from scipy.stats import chisquare, chi2
		from matplotlib.offsetbox import AnchoredOffsetbox, TextArea


		def safe_exp(x, min_exp_arg=None, max_exp_arg=None):
			min_exp_arg = min_exp_arg if min_exp_arg is not None else -np.inf
			max_exp_arg = max_exp_arg if max_exp_arg is not None else np.finfo(np.float64).maxexp - 10
			return np.exp(np.clip(x, min_exp_arg, max_exp_arg))

		
		def emg_stable(x, amplitude, mu, sigma, lambda_):
			exp_arg = 0.5 * lambda_ * (2 * mu + lambda_ * sigma**2 - 2 * x)
			erfc_arg = (mu + lambda_ * sigma**2 - x) / (np.sqrt(2) * sigma)
			#print("lambda_: ", lambda_)
			#print("mu: ", mu)
			#print("sigma: ", sigma)
			#print("x: ", x)
			return 0.5 * amplitude * lambda_ * safe_exp(exp_arg - erfc_arg**2) * erfcx(erfc_arg)

		def composite_emg(x, *params):
			result = np.zeros_like(x)
			for i in range(0, len(params), 4):
				result += emg_stable(x, *params[i:i + 4])
			return result


		# Set the threshold for y_hist, adjust it based on your specific requirements
		y_hist_threshold = 1e5

		# Filter the data based on the threshold
		valid_indices = y_hist < y_hist_threshold
		filtered_x_hist = x_hist[valid_indices]
		filtered_y_hist = y_hist[valid_indices]


		# Fit the composite EMG function to the data
		popt, pcov = curve_fit(composite_emg, filtered_x_hist, filtered_y_hist, p0=peak_info, maxfev=1000000)
		fitted_emg = composite_emg(filtered_x_hist, *popt)
		#print('FINAL FIT PARAMETERS:', [*popt])
		# Print final fit parameters
		print("FINAL FIT PARAMETERS:")
		for i in range(0, len(popt), 4):
			print(f"Peak {i//4 + 1}: Amp={peak_info[i]}, Mu={peak_info[i+1]}, Sigma={peak_info[i+2]}, Lambda={peak_info[i+3]}")

		# Print final fit parameters
		def display_fit_parameters(peak_info, popt, fixed_list=None):
			fit_params_window = Toplevel()
			fit_params_window.title("Final Fit Parameters")
			fit_params_window.geometry("700x200")
			output_text = Text(fit_params_window, wrap=WORD)
			output_text.pack(expand=True, fill=BOTH)

			param_names = ['Amp', 'Mu', 'Sigma', 'Lambda']
			idx = 0
			fixed_param_idx = 0

			# If fixed_list is not provided, create a list of all False values
			if fixed_list is None:
				fixed_list = [False] * len(peak_info)

			for i in range(0, len(peak_info), 4):
				peak_label = f"Peak {(i // 4) + 1}: "
				for j in range(4):
					if fixed_list[i + j]:
						peak_label += f"*{param_names[j]}={fixed_params[fixed_param_idx]}, "
						fixed_param_idx += 1
					else:
						peak_label += f"{param_names[j]}={popt[idx]}, "
						idx += 1
				output_text.insert(END, peak_label + "\n")


		display_fit_parameters(peak_info, popt)

		# Calibration points
		calib_point_1 = (0.806, 156745)
		calib_point_2 = (1.679, 320842)
		#calib_point_1 = (0.806, 157600)
		#calib_point_2 = (1.679, 275300)
		#calib_point_1 = (0.303, 84672)
		#calib_point_2 = (2.150, 374439)
		#calib_point_1 = (0.806, 157700)
		#calib_point_2 = (1.679, 308200)

		# Convert calibration points to MeV
		energy_1, channel_1 = calib_point_1
		energy_2, channel_2 = calib_point_2
		energy_scale_factor = (energy_2 - energy_1) / (channel_2 - channel_1)
		energy_offset = energy_1 - energy_scale_factor * channel_1

		# Convert filtered_x_hist and tot_energy to MeV
		filtered_x_hist_MeV = filtered_x_hist * energy_scale_factor + energy_offset
		tot_energy_MeV = np.array(tot_energy) * energy_scale_factor + energy_offset

		# Plot the histogram and the fitted EMG on the main plot (ax1) with the calibrated x-axis
		plt.hist(tot_energy_MeV, bins=num_bins)
		plt.rcParams['figure.figsize'] = [10, 10]
		plt.title(f'Energy Spectrum', fontdict = {'fontsize' : 20})
		plt.xlabel(f'Energy (MeV)', fontdict = {'fontsize' : 20})  # Update the x-axis label
		plt.ylabel(f'Counts',fontdict = {'fontsize' : 20})
		plt.plot(filtered_x_hist_MeV, fitted_emg, linestyle='--', label='Multi EMG Fit')  # Use the calibrated x-axis values

		# Convert mu integrated charge to energy
		mu_values = [mu * energy_scale_factor + energy_offset for mu in popt[1::4]]

		# Add peak labels
		for idx, mu_value in enumerate(mu_values):
			y_value = fitted_emg[np.argmin(np.abs(filtered_x_hist_MeV - mu_value))]
			plt.annotate(f"< {mu_value:.2f} MeV | Peak {idx+1}", (mu_value, y_value),textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, color='black', rotation=90)

		plt.ylim(0, 1.55 * np.max(fitted_emg))
		plt.show()



	def parse_peak_info_string(peak_info_str):
		param_pattern = r"(\*?[A-Za-z]+)=([-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?)"
		param_regex = re.compile(param_pattern)

		peak_lines = peak_info_str.split("\n")

		float_list = []
		fixed_list = []
		for line in peak_lines:
			params = param_regex.findall(line)

			for param, value in params:
				float_list.append(float(value.replace(",", "")))
				fixed_list.append(param.startswith("*"))

		return float_list, fixed_list


	def plot_spectrum_multi_params():
		from scipy.special import erfc
		from scipy.special import erfcx
		from scipy.optimize import curve_fit
		from scipy.stats import chisquare, chi2
		from matplotlib.offsetbox import AnchoredOffsetbox, TextArea


		def safe_exp(x, min_exp_arg=None, max_exp_arg=None):
			min_exp_arg = min_exp_arg if min_exp_arg is not None else -np.inf
			max_exp_arg = max_exp_arg if max_exp_arg is not None else np.finfo(np.float64).maxexp - 10
			return np.exp(np.clip(x, min_exp_arg, max_exp_arg))

		
		def emg_stable(x, amplitude, mu, sigma, lambda_):
			exp_arg = 0.5 * lambda_ * (2 * mu + lambda_ * sigma**2 - 2 * x)
			erfc_arg = (mu + lambda_ * sigma**2 - x) / (np.sqrt(2) * sigma)
			#print("lambda_: ", lambda_)
			#print("mu: ", mu)
			#print("sigma: ", sigma)
			#print("x: ", x)
			return 0.5 * amplitude * lambda_ * safe_exp(exp_arg - erfc_arg**2) * erfcx(erfc_arg)

		def composite_emg(x, params, fixed_params):
			print("params:", params)
			print("fixed_params:", fixed_params)
			print("fixed_list:", fixed_list)
			result = np.zeros_like(x)
			fixed_param_idx = 0
			param_idx = 0
			for i in range(0, len(params) + len(fixed_params), 4):
				current_params = [0] * 4
				for j in range(4):
					if fixed_list[i + j]:
						current_params[j] = fixed_params[fixed_param_idx]
						fixed_param_idx += 1
					else:
						current_params[j] = params[param_idx]
						param_idx += 1
				result += emg_stable(x, *current_params)
			return result


		# Get peak info, bins, and hist data
		num_bins = int(entry_binsRvE.get())
		y_hist, x_hist = np.histogram(tot_energy, bins=num_bins)
		x_hist = (x_hist[:-1] + x_hist[1:]) / 2

		# Set the threshold for y_hist, adjust it based on your specific requirements
		y_hist_threshold = 1e5

		# Filter the data based on the threshold
		valid_indices = y_hist < y_hist_threshold
		filtered_x_hist = x_hist[valid_indices]
		filtered_y_hist = y_hist[valid_indices]

		# Get peak info, bins, and hist data
		input_str = entry_multi.get()
		peak_info, fixed_list = parse_peak_info_string(input_str)
		#print("Input string:", input_str)
		#print("Parsed peak_info:", peak_info)
		#print("Parsed fixed_list:", fixed_list)

		# Filter out fixed parameters from the initial guess for the curve_fit function
		initial_guess = [peak_info[i] for i in range(len(peak_info)) if not fixed_list[i]]

		# Fit the composite EMG function to the data
		fixed_params = [peak_info[i] for i in range(len(peak_info)) if fixed_list[i]]
		popt, pcov = curve_fit(lambda x, *params: composite_emg(x, params, fixed_params), filtered_x_hist, filtered_y_hist, p0=initial_guess, maxfev=1000000)

		fitted_emg = composite_emg(filtered_x_hist, popt, fixed_params)


		# Print final fit parameters
		print("FINAL FIT PARAMETERS:")
		param_names = ['Amp', 'Mu', 'Sigma', 'Lambda']
		idx = 0
		fixed_param_idx = 0

		for i in range(0, len(peak_info), 4):
			print(f"Peak {(i // 4) + 1}:", end=" ")
			for j in range(4):
				if fixed_list[i + j]:
					print(f"*{param_names[j]}={fixed_params[fixed_param_idx]}", end=", ")
					fixed_param_idx += 1
				else:
					print(f"{param_names[j]}={popt[idx]}", end=", ")
					idx += 1
			print()

		# Print final fit parameters
		def display_fit_parameters(peak_info, popt, fixed_list=None):
			fit_params_window = Toplevel()
			fit_params_window.title("Final Fit Parameters")
			fit_params_window.geometry("700x200")
			output_text = Text(fit_params_window, wrap=WORD)
			output_text.pack(expand=True, fill=BOTH)

			param_names = ['Amp', 'Mu', 'Sigma', 'Lambda']
			idx = 0
			fixed_param_idx = 0

			# If fixed_list is not provided, create a list of all False values
			if fixed_list is None:
				fixed_list = [False] * len(peak_info)

			for i in range(0, len(peak_info), 4):
				peak_label = f"Peak {(i // 4) + 1}: "
				for j in range(4):
					if fixed_list[i + j]:
						peak_label += f"*{param_names[j]}={fixed_params[fixed_param_idx]}, "
						fixed_param_idx += 1
					else:
						peak_label += f"{param_names[j]}={popt[idx]}, "
						idx += 1
				output_text.insert(END, peak_label + "\n")


		display_fit_parameters(peak_info, popt, fixed_list)

		# Calculate the chi-squared value
		chi_squared = np.sum(((filtered_y_hist - fitted_emg) ** 2) / fitted_emg)
		degrees_of_freedom = len(filtered_y_hist) - len(popt)
		chi_squared_per_dof = chi_squared / degrees_of_freedom

		# Calculate the p-value
		p_value = 1 - chi2.cdf(chi_squared, degrees_of_freedom)

		# Calibration points
		calib_point_1 = (0.806, 156745)
		calib_point_2 = (1.679, 320842)
		#calib_point_1 = (0.806, 157600)
		#calib_point_2 = (1.679, 275300)
		#calib_point_1 = (0.303, 84672)
		#calib_point_2 = (2.150, 374439)
		#calib_point_1 = (0.806, 157700)
		#calib_point_2 = (1.679, 308200)

		# Convert calibration points to MeV
		energy_1, channel_1 = calib_point_1
		energy_2, channel_2 = calib_point_2
		energy_scale_factor = (energy_2 - energy_1) / (channel_2 - channel_1)
		energy_offset = energy_1 - energy_scale_factor * channel_1

		# Convert filtered_x_hist and tot_energy to MeV
		filtered_x_hist_MeV = filtered_x_hist * energy_scale_factor + energy_offset
		tot_energy_MeV = np.array(tot_energy) * energy_scale_factor + energy_offset
		

		# Plot the smooth histogram and the fitted EMG on the main plot (ax1)
		#plt.plot(x_hist, y_hist, label='Histogram')
		plt.hist(tot_energy_MeV, bins=num_bins)
		plt.rcParams['figure.figsize'] = [10, 10]
		plt.title(f'Energy Spectrum', fontdict = {'fontsize' : 20})
		plt.xlabel(f'Energy (MeV)', fontdict = {'fontsize' : 20})
		plt.ylabel(f'Counts',fontdict = {'fontsize' : 20})
		plt.plot(filtered_x_hist_MeV, fitted_emg, linestyle='--', label='Multi EMG Fit')

		# Convert mu integrated charge to energy
		mu_values = [mu * energy_scale_factor + energy_offset for mu in popt[1::4]]

		# Add peak labels
		for idx, mu_value in enumerate(mu_values):
			y_value = fitted_emg[np.argmin(np.abs(filtered_x_hist_MeV - mu_value))]
			plt.annotate(f"< {mu_value:.2f} MeV | Peak {idx+1}", (mu_value, y_value),textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, color='black', rotation=90)

		plt.ylim(0, 1.55 * np.max(fitted_emg))
		plt.show()


	def plot_spectrum_fit_gauss():
		# Get the number of bins from the entry widget
		num_bins1 = int(entry_binsRvE.get())
		low_cut_value = int(entry_low.get())
		high_cut_value = int(entry_high.get())

		# Get proportional bin width 
		num_bins = int(((high_cut_value - low_cut_value) / np.max(tot_energy)) * num_bins1)
		hist, bins = np.histogram(tot_energy, bins=num_bins, range=(low_cut_value, high_cut_value))
		bin_centers = (bins[:-1] + bins[1:]) / 2

		def gaussian(x, amplitude, mu, sigma):
			return amplitude * np.exp(-(x - mu)**2 / (2 * sigma**2))

		popt, pcov = curve_fit(gaussian, bin_centers, hist, p0=[1, np.mean(tot_energy), np.std(tot_energy)], maxfev=800000)
		amplitude, mu, sigma = popt

		# Calculate chi-squared and p-value
		residuals = hist - gaussian(bin_centers, amplitude, mu, sigma)
		ss_res = np.sum(residuals**2)
		ss_tot = np.sum((hist - np.mean(hist))**2)
		r_squared = 1 - (ss_res / ss_tot)
		chi_squared = ss_res / (num_bins - 3)
		dof = num_bins - 3
		chi_squared_dof = chi_squared / dof
		p_value = 1 - chi2.cdf(chi_squared, dof)

		# Plot the histogram with the fit
		fig, ax = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
		ax[0].hist(tot_energy, bins=num_bins, range=(low_cut_value, high_cut_value), histtype='step', color='blue', label='Data')
		x_fit = np.linspace(low_cut_value, high_cut_value, 100)
		ax[0].plot(x_fit, gaussian(x_fit, amplitude, mu, sigma), 'r-', label='Fit')
		ax[0].legend()
		ax[0].set_ylabel('Counts')

		# Plot the residuals
		ax[1].plot(bin_centers, residuals, 'b-', label='Residuals')
		ax[1].axhline(0, color='black', lw=1)
		ax[1].set_xlabel('Energy')
		ax[1].set_ylabel('Residuals')
		ax[1].legend()
		plt.tight_layout()

		# Display the fit parameters on the plot
		text = f'Chi-squared: {chi_squared:.2f}\nDegrees of Freedom: {dof}\nChi-squared per DOF: {chi_squared_dof:.2f}\np-value: {p_value:.2f}\nAmplitude: {amplitude:.2f}\nMu: {mu:.2f}\nSigma: {sigma:.2f}'
		ax[0].text(0.05, 0.95, text, transform=ax[0].transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

		plt.show()


		return 
	


	def temp_text4(e):
		entry_low.delete(0,"end")

	def temp_text5(e):
		entry_high.delete(0,"end")

	def temp_text6(e):
		entry_binsRvE.delete(0,"end")

	def temp_text7(e):
		entry_bins_eng_spec.delete(0,"end")

	def temp_text8(e):
		entry_binsRvE2.delete(0,"end")

	def temp_text9(e):
		entry_low3.delete(0,"end")

	def temp_text10(e):
		entry_high3.delete(0,"end")

	def temp_text11(e):
		entry_binsRvE3.delete(0,"end")
	
	def temp_textamp(e):
		entry_amp.delete(0,"end")

	def temp_textmu(e):
		entry_mu.delete(0,"end")

	def temp_textsigma(e):
		entry_sigma.delete(0,"end")

	def temp_textlambda(e):
		entry_lambda.delete(0,"end")

	def temp_text61(e):
		entry_multi.delete(0,"end")


	def switch_eng_spec():
		global is_on_eng_spec

		# Determine is on or off
		if is_on_eng_spec:
			is_on_eng_spec = False
			entry_binsRvE.destroy()
			button_spec.destroy()
			button_eng_spec.configure(fg='red')

			
		else:
			is_on_eng_spec = True
			button_eng_spec.configure(fg='green')
			open_eng_spec()


	def apply_cut():
		low_cut = int(entry_low.get())
		high_cut = int(entry_high.get())
		num_bins = int(entry_binsRvE.get())
		trimmed_hist = tot_energy
		trimmed_hist = [x for x in trimmed_hist if x <= high_cut]
		trimmed_hist = [x for x in trimmed_hist if x >= low_cut]
		# print(f'Number of Counts: {len(trimmed_hist)}')
		n, bins, patches = plt.hist(tot_energy, num_bins, facecolor='green', alpha=0.75)
		#hist, bins = np.histogram(tot_energy, bins=num_bins, range=(low_cut, high_cut))
		plt.xlabel('Integrated Charge', fontdict = {'fontsize' : 20})
		plt.ylabel('Counts',  fontdict = {'fontsize' : 20})
		plt.axvline(low_cut, color='red', linestyle='dashed', linewidth=2)
		plt.axvline(high_cut, color='red', linestyle='dashed', linewidth=2)
		plt.title(f'Number of Counts in Cut: {len(trimmed_hist)}',fontdict = {'fontsize' : 20})
		plt.grid(True)
		plot = plt.show()
		return plot


	def switch_peak():
		global is_on_peak

		# Determine is on or off
		if is_on_peak:
			is_on_peak = False
			entry_low.destroy()
			entry_high.destroy()
			button_apply.destroy()
			entry_binsRvE.destroy()
			button_spec.destroy()

			#canvas1.create_window(163, 260, window=button_peak)

			
		else:
			is_on_peak = True
			#button_peak = Button(root, text="Peak Fitting", fg='green')
			#canvas1.create_window(163, 260, window=button_peak)
			open_peak_fit()

	def close_out():
		entry_low.destroy()
		entry_high.destroy()
		button_apply.destroy()
		entry_binsRvE.destroy()
		button_spec.destroy()
		button_eng_spec_temp.destroy()
		button_3D.destroy()
		button_track_angle.destroy()
		button_cnn.destroy()
		button_track_trace.destroy()
		button_RvE.destroy()
		#entry_amp.destroy()
		#entry_mu.destroy()
		#entry_sigma.destroy()
		#entry_lambda.destroy()
		button_fit.destroy()
		button_fit_multi.destroy()
		entry_multi.destroy()
		button_fit_multi_params.destroy()

	def on_focus_in(event):
		if event.widget.get() == event.widget.default_text:
			event.widget.delete(0, END)

	def on_focus_out(event):
		if event.widget.get() == '':
			event.widget.insert(0, event.widget.default_text)

	entry_binsRvE = Entry(root, borderwidth=5, width=12)
	entry_binsRvE.default_text = 'Enter # of Bins'
	entry_binsRvE.insert(0, entry_binsRvE.default_text)
	entry_binsRvE.bind('<FocusIn>', on_focus_in)
	entry_binsRvE.bind('<FocusOut>', on_focus_out)
	canvas1.create_window(480, 135, window=entry_binsRvE)

	button_spec = Button(root, text="Generate Spectrum", command=plot_spectrum)
	canvas1.create_window(480, 205, window=button_spec)

	entry_low = Entry(root, width=11) 
	entry_low.default_text = 'Low Cut Value'
	entry_low.insert(0, entry_low.default_text)
	entry_low.bind('<FocusIn>', on_focus_in)
	entry_low.bind('<FocusOut>', on_focus_out)
	canvas1.create_window(440, 265, window=entry_low)

	entry_high = Entry(root, width=11) 
	entry_high.default_text = 'High Cut Value'
	entry_high.insert(0, entry_high.default_text)
	entry_high.bind('<FocusIn>', on_focus_in)
	entry_high.bind('<FocusOut>', on_focus_out)
	canvas1.create_window(525, 265, window=entry_high)

	button_apply = Button(root, text="Apply Cut Range", command=apply_cut)
	canvas1.create_window(480, 295, window=button_apply)

	#entry_amp = Entry(root, width=11) 
	#entry_amp.insert(0, "Amplitude")
	#entry_amp.bind("<FocusIn>", temp_textamp)
	#canvas1.create_window(350, 385, window=entry_amp)

	#entry_mu = Entry(root, width=11) 
	#entry_mu.insert(0, "Mu")
	#entry_mu.bind("<FocusIn>", temp_textmu)
	#canvas1.create_window(435, 385, window=entry_mu)

	#entry_sigma = Entry(root, width=11) 
	#entry_sigma.insert(0, "Sigma")
	#entry_sigma.bind("<FocusIn>", temp_textsigma)
	#canvas1.create_window(520, 385, window=entry_sigma)

	#entry_lambda = Entry(root, width=11) 
	#entry_lambda.insert(0, "Lambda")
	#entry_lambda.bind("<FocusIn>", temp_textlambda)
	#canvas1.create_window(605, 385, window=entry_lambda)

	button_fit = Button(root, text="Quick Gaussian Fit", command=plot_spectrum_fit_gauss)
	canvas1.create_window(480, 325, window=button_fit)

	button_fit_multi = Button(root, text="Initial Guesses for Multi-peak Fit", command=plot_spectrum_multi)
	canvas1.create_window(480, 385, window=button_fit_multi)

	entry_multi = Entry(root, width=42) 
	entry_multi.insert(0,'Paste Fit Parameters | Use * in Front of Param to Fix Value')
	entry_multi.bind('<FocusIn>', temp_text61)
	canvas1.create_window(480, 445, window=entry_multi)


	button_fit_multi_params = Button(root, text="Multi-peak Fit from Params", command=plot_spectrum_multi_params)
	canvas1.create_window(480, 475, window=button_fit_multi_params)

	#button_fit2 = Button(root, text="Fit w/ Emg", command=plot_spectrum_fit_emg)
	#canvas1.create_window(480, 415, window=button_fit2)

	# Energy Spec Button	
	button_eng_spec_temp = Button(text='Energy Spectrum',fg='red', command=close_out)
	canvas1.create_window(163, 235, window=button_eng_spec_temp)

	# Range vs Energy Button
	button_RvE = Button(text='Range vs Energy',fg='green',command=open_RvE)
	canvas1.create_window(163, 285, window=button_RvE)
	button_RvE["state"] = "disabled"

	# 3D Plot Button
	button_3D = Button(text='3D Event Plot',fg='green', command=open_3d_plot)
	canvas1.create_window(163, 335, window=button_3D)
	button_3D["state"] = "disabled"

	# Track with Trace Button
	button_track_trace = Button(text='Track with Trace',fg='green', command=open_track_trace)
	canvas1.create_window(163, 385, window=button_track_trace)
	button_track_trace["state"] = "disabled"

	# Track Angle Button
	button_track_angle = Button(text='Track Angles',fg='green', command=open_track_angles)
	canvas1.create_window(163, 435, window=button_track_angle)
	button_track_angle["state"] = "disabled"

	# ConvNet Button
	button_cnn = Button(text='ConvNet Track ID',fg='green', command=cnn)
	canvas1.create_window(163, 485, window=button_cnn)
	button_cnn["state"] = "disabled"

	

############################################################## Start
##############################################################

def start(run_num):

	def temp_text4(e):
		entry_low.delete(0,"end")

	def temp_text5(e):
		entry_veto_intCharge.delete(0,"end")

	def temp_text7(e):
		entry_veto_length.delete(0,"end")

	def temp_text8(e):
		entry_pad_threshold.delete(0,"end")


	def create_files():

		# Use np.save all arrays to directory /mnt/analysis/e21072/analysis/files/run_{run_num}_param1_param2....
		# Each array is saved with the same name but under the specific directory with the specific parameters

		length = int(entry_veto_length.get())
		ic = int(entry_veto_intCharge.get()) * 10**5
		pads = int(entry_pad_threshold.get())
		eps = int(entry_eps.get())
		samps = int(entry_min_samps.get())
		poly = int(entry_poly.get())

		# mypath = f"/mnt/analysis/e21072/analysis_files/run_{run_num}"
		mypath = f"/mnt/analysis/e21072/h5test/run_{run_num}"
		sub_mypath = f"/mnt/analysis/e21072/h5test/run_{run_num}/len{length}_ic{ic}_pads{pads}_eps{eps}_samps{samps}_poly{poly}"

		if os.path.isdir(mypath):
			if os.path.isdir(sub_mypath):

				# Give option to overwrite or cancel
				print("Files Already Exist.")
				overwrite = int(input('Would you like to overwrite (1=yes, 0=no): '))	


				if overwrite == True:
					print('Overwriting Existing Files')
				else:

					entry_veto_length.destroy()
					label_veto_length.destroy()

					entry_veto_intCharge.destroy()
					label_veto_intCharge.destroy()

					entry_pad_threshold.destroy()
					label_pad_threshold.destroy()

					entry_eps.destroy()
					label_eps.destroy()

					entry_min_samps.destroy()
					label_min_samps.destroy()

					entry_poly.destroy()
					label_poly.destroy()
					

					button_create.destroy()
					return
				
			else:
				os.makedirs(sub_mypath)
			
		else:
			os.makedirs(mypath)
			os.makedirs(sub_mypath)

		# Run main with the given parameters
		generate_files(run_num, length, ic, pads, eps, samps, poly)

		entry_veto_length.destroy()
		label_veto_length.destroy()

		entry_veto_intCharge.destroy()
		label_veto_intCharge.destroy()

		entry_pad_threshold.destroy()
		label_pad_threshold.destroy()

		entry_eps.destroy()
		label_eps.destroy()

		entry_min_samps.destroy()
		label_min_samps.destroy()

		entry_poly.destroy()
		label_poly.destroy()
		

		button_create.destroy()
	


	entry_veto_length = Entry(root, width=4) 
	entry_veto_length.insert(0, 80)
	entry_veto_length.bind("<FocusIn>", temp_text7)
	canvas1.create_window(280, 145, height=21, window=entry_veto_length)

	label_veto_length = Label(root, text="(mm)                     Length Veto",bg="white", fg="black", font=('helvetica','10'), relief="groove")
	label_veto_length.place(x=305, y=136)

	
	entry_veto_intCharge = Entry(root, width=4) 
	entry_veto_intCharge.insert(0, 8)
	entry_veto_intCharge.bind("<FocusIn>", temp_text5)
	canvas1.create_window(280, 175, height=21, window=entry_veto_intCharge)

	label_veto_intCharge = Label(root, text="(x10e5)                 Integrated Charge Veto",bg="white", fg="black", font=('helvetica','10'), relief="groove") 
	label_veto_intCharge.place(x=305, y=166)


	entry_pad_threshold = Entry(root, width=4) 
	entry_pad_threshold.insert(0, 21)
	entry_pad_threshold.bind("<FocusIn>", temp_text8)
	canvas1.create_window(280, 205, height=21, window=entry_pad_threshold)

	label_pad_threshold = Label(root, text="(points)                 Points Threshold",bg="white", fg="black", font=('helvetica','10'), relief="groove")
	label_pad_threshold.place(x=305, y=196)


	entry_eps = Entry(root, width=4) 
	entry_eps.insert(0, 7)
	entry_eps.bind("<FocusIn>", temp_text5)
	canvas1.create_window(280, 235, height=21, window=entry_eps)

	label_eps = Label(root, text="(eps)                     DBSCAN Parm",bg="white", fg="black", font=('helvetica','10'), relief="groove")
	label_eps.place(x=305, y=226)


	entry_min_samps = Entry(root, width=4) 
	entry_min_samps.insert(0, 8)
	entry_min_samps.bind("<FocusIn>", temp_text5)
	canvas1.create_window(280, 265, height=21, window=entry_min_samps)

	label_min_samps = Label(root, text="(min samples)       DBSCAN Parm",bg="white", fg="black", font=('helvetica','10'), relief="groove")
	label_min_samps.place(x=305, y=256)


	entry_poly = Entry(root, width=4) 
	entry_poly.insert(0, 2)
	entry_poly.bind("<FocusIn>", temp_text5)
	canvas1.create_window(280, 295, height=21, window=entry_poly)

	label_poly = Label(root, text="(poly degree)        IMOD-Poly Parm",bg="white", fg="black", font=('helvetica','10'), relief="groove")
	label_poly.place(x=305, y=286)

	button_create = Button(text='CREATE FILES', borderwidth=5, fg='green', command=create_files)
	canvas1.create_window(163, 170, window=button_create)

	return
	

def find(run_num):
	
	#mypath = f"/mnt/analysis/e21072/analysis_files/run_{run_num}"
	mypath = f"/mnt/analysis/e21072/h5test/run_{run_num}"
	global sub_mymainpath
	sub_mymainpath = filedialog.askdirectory(initialdir = mypath, title = "Select a Directory")

	global tot_energy
	tot_energy = np.load(os.path.join(sub_mymainpath, 'tot_energy.npy'), allow_pickle=True)

	global skipped_events
	skipped_events = np.load(os.path.join(sub_mymainpath, 'skipped_events.npy'), allow_pickle=True)

	global veto_events
	veto_events = np.load(os.path.join(sub_mymainpath, 'veto_events.npy'), allow_pickle=True)

	global good_events
	good_events = np.load(os.path.join(sub_mymainpath, 'good_events.npy'), allow_pickle=True)

	global len_list
	len_list = np.load(os.path.join(sub_mymainpath, 'len_list.npy'), allow_pickle=True)

	global trace_list
	trace_list = np.load(os.path.join(sub_mymainpath, 'trace_list.npy'), allow_pickle=True)

	global angle_list
	angle_list = np.load(os.path.join(sub_mymainpath, 'angle_list.npy'), allow_pickle=True)


	# Enable Buttons 
	button_eng_spec["state"] = "normal"
	button_RvE["state"] = "normal"
	button_3D["state"] = "normal"
	button_track_trace["state"] = "normal"
	button_track_angle["state"] = "normal"
	button_cnn["state"] = "normal"

	return 



############################################################### 3D Plot
###############################################################

def plot_3D(run_num):
	

	def temp_text7(e):
		entry_event_num.delete(0,"end")

	def plot_track():
		############ We need to grab the could points here not the hit list....
		orig_num = int(entry_event_num.get())
		event_num = np.where(good_events == orig_num)[0][0]
		xHit_list = np.load(os.path.join(sub_mymainpath, 'xHit_list.npy'), allow_pickle=True)
		yHit_list = np.load(os.path.join(sub_mymainpath, 'yHit_list.npy'), allow_pickle=True)
		zHit_list = np.load(os.path.join(sub_mymainpath, 'zHit_list.npy'), allow_pickle=True)
		eHit_list = np.load(os.path.join(sub_mymainpath, 'eHit_list.npy'), allow_pickle=True)

		xHit = xHit_list[event_num]
		yHit = yHit_list[event_num]
		zHit = zHit_list[event_num]
		eHit = eHit_list[event_num]

		fig = plt.figure(figsize=(6,6))
		ax = plt.axes(projection='3d')
		ax.set_xlim3d(-35, 35)
		ax.set_ylim3d(-35, 35)
		ax.set_zlim3d(0, 35)
		ax.set_xlabel("x")
		ax.set_ylabel("y")
		ax.set_zlabel("z")
		ax.set_title(f"3D Point-cloud of Event {orig_num}\nLength: {len_list[event_num]:.2f}\nAngle: {angle_list[event_num]:.2f}", fontdict = {'fontsize' : 10})
		ax.scatter(xHit, yHit, zHit-np.min(zHit), c=eHit, cmap='RdBu_r')
		cbar = fig.colorbar(ax.get_children()[0])
		plt.show() 


	def plot_track_dense():
		from sklearn.neighbors import NearestNeighbors
		radius = 5
		orig_num = int(entry_event_num.get())
		event_num = np.where(good_events == orig_num)[0][0]
		xHit_list = np.load(os.path.join(sub_mymainpath, 'xHit_list.npy'), allow_pickle=True)
		yHit_list = np.load(os.path.join(sub_mymainpath, 'yHit_list.npy'), allow_pickle=True)
		zHit_list = np.load(os.path.join(sub_mymainpath, 'zHit_list.npy'), allow_pickle=True)
		eHit_list = np.load(os.path.join(sub_mymainpath, 'eHit_list.npy'), allow_pickle=True)

		xHit = xHit_list[event_num]
		yHit = yHit_list[event_num]
		zHit = zHit_list[event_num]
		eHit = eHit_list[event_num]

		# Define the nearest neighbors algorithm
		nbrs = NearestNeighbors(radius=radius).fit(np.vstack((xHit, yHit, zHit)).T)

		# Find the points within the specified radius
		points_within_radius = nbrs.radius_neighbors(np.vstack((xHit, yHit, zHit)).T, return_distance=False)

		# Interpolate between points within the radius
		xHit_dense, yHit_dense, zHit_dense, eHit_dense = [], [], [], []
		for i, neighbors in enumerate(points_within_radius):
			for j in neighbors:
				t = np.random.rand()
				xHit_dense.append(xHit[i] + t * (xHit[j] - xHit[i]))
				yHit_dense.append(yHit[i] + t * (yHit[j] - yHit[i]))
				zHit_dense.append(zHit[i] + t * (zHit[j] - zHit[i]))
				eHit_dense.append(eHit[i] + t * (eHit[j] - eHit[i]))

		# Convert lists to arrays
		xHit_dense = np.array(xHit_dense)
		yHit_dense = np.array(yHit_dense)
		zHit_dense = np.array(zHit_dense)
		eHit_dense = np.array(eHit_dense)

		# Plot the dense point cloud
		fig = plt.figure(figsize=(6, 6))
		ax = plt.axes(projection='3d')
		ax.set_xlim3d(-35, 35)
		ax.set_ylim3d(-35, 35)
		ax.set_zlim3d(0, 35)
		ax.set_xlabel("x")
		ax.set_ylabel("y")
		ax.set_zlabel("z")
		ax.set_title(f"Dense 3D Point-cloud of Event {orig_num}\nLength: {len_list[event_num]:.2f} mm\nAngle: {angle_list[event_num]:.2f} degrees", fontdict = {'fontsize' : 10})
		ax.scatter(xHit_dense, yHit_dense, zHit_dense - np.min(zHit_dense), c=eHit_dense, cmap='RdBu_r')
		cbar = fig.colorbar(ax.get_children()[0])
		cbar.set_label('eHit')

		plt.show()


	def close_out():
		entry_event_num.destroy()
		button_plot.destroy()
		button_3D.destroy()
		button_eng_spec.destroy()
		button_RvE.destroy()
		button_track_angle.destroy()
		button_cnn.destroy()
		button_track_trace.destroy()
		button_plot_dense.destroy()

	def on_focus_in(event):
		if event.widget.get() == event.widget.default_text:
			event.widget.delete(0, END)

	def on_focus_out(event):
		if event.widget.get() == '':
			event.widget.insert(0, event.widget.default_text)


	entry_event_num = Entry(root, borderwidth=5, width=11) 
	entry_event_num.default_text = 'Enter Event #'
	entry_event_num.insert(0, entry_event_num.default_text)
	entry_event_num.bind("<FocusIn>", on_focus_in)
	entry_event_num.bind("<FocusOut>", on_focus_out)
	canvas1.create_window(480, 270, window=entry_event_num)

	button_plot = Button(root, text="Plot 3D Track", command=plot_track)
	canvas1.create_window(480, 320, window=button_plot)

	button_plot_dense = Button(root, text="Plot Dense 3D Track", command=plot_track_dense)
	canvas1.create_window(480, 350, window=button_plot_dense)

	button_3D = Button(text='3D Event Plot',fg='red', command=close_out)
	canvas1.create_window(163, 335, window=button_3D)

	# Energy Spectrum Button    
	button_eng_spec = Button(text='Energy Spectrum',fg='green', command=open_eng_spec)
	canvas1.create_window(163, 235, window=button_eng_spec)
	button_eng_spec["state"] = "disabled"

	# Range vs Energy Button
	button_RvE = Button(text='Range vs Energy',fg='green',command=open_RvE)
	canvas1.create_window(163, 285, window=button_RvE)
	button_RvE["state"] = "disabled"

	# Track with Trace Button
	button_track_trace = Button(text='Track with Trace',fg='green', command=open_track_trace)
	canvas1.create_window(163, 385, window=button_track_trace)
	button_track_trace["state"] = "disabled"

	# Track Angle Button
	button_track_angle = Button(text='Track Angles',fg='green', command=open_track_angles)
	canvas1.create_window(163, 435, window=button_track_angle)
	button_track_angle["state"] = "disabled"

	# ConvNet Button
	button_cnn = Button(text='ConvNet Track ID',fg='green', command=cnn)
	canvas1.create_window(163, 485, window=button_cnn)
	button_cnn["state"] = "disabled"



############################################################### Track with Traces
###############################################################

def track_trace(run_num):

	
	def make_grid():
		"""
		"Create Training Data.ipynb"eate grid matrix of MM outline and energy bar, see spreadsheet below
		https://docs.google.com/spreadsheets/d/1_bbg6svfEph_g_Z002rmzTLu8yjQzuj_p50wqs7mMrI/edit?usp=sharing
		"""
		row = np.array([63, 47, 39, 31, 27, 23, 19, 15, 15, 11, 11, 7, 7, 7, 7, 3, 3, 3, 3, 3, 3, 3, 7, 7, 
			    7, 7, 11, 11, 15, 15, 19, 23, 27, 31, 39, 47, 63]) 

		to_row = np.array([87, 103, 111, 119, 123, 127, 131, 135, 135, 139, 139, 143, 143, 143, 143, 147, 
			       147, 147, 147, 147, 147, 148, 143, 143, 143, 144, 139, 140, 135, 136, 132, 128,
			       124, 120, 112, 104, 88]) 

		col = np.array([100, 84, 76, 68, 64, 60, 56, 52, 52, 48, 48, 44, 44, 44, 44, 40, 40, 40, 40, 40, 
			    40, 40, 44, 44, 44, 44, 48, 48, 52, 52, 56, 60, 64, 68, 76, 84, 100])

		to_col = np.array([124, 140, 148, 156, 160, 164, 168, 172, 172, 176, 176, 180, 180, 180, 180, 184, 
			       184, 184, 184, 184, 184, 184, 180, 180, 180, 180, 176, 176, 172, 172, 168, 164, 
			       160, 156, 148, 140, 124]) 

		all_row = np.array([i for i in range(3, 148, 4)])
		all_col = np.array([i for i in range(40, 185, 4)]) 

		full_image_size_width = 224
		full_image_size_length = 151
		mm_grid = np.zeros((full_image_size_length, full_image_size_width, 3))    
		mm_grid.fill(255)                                                     

		for i in range(len(row)):
			# draw grid columns, 0 = black
			mm_grid[row[i]:to_row[i], all_col[i], :] = 0
        
		for i in range(len(col)):
			# draw grid rows
			mm_grid[all_row[i], col[i]:to_col[i], :] = 0

		# Ensure that matrix is of integers
		mm_grid = mm_grid.astype(int) 

		# Draw engergy bar box
		mm_grid = make_box(mm_grid)

		return mm_grid


	def blue_range(pad_plane, rows):

		start_row = 140
		low_color = 0
		high_color = 35

		for i in range(rows):
			pad_plane[start_row:start_row+5, 8:17, 0] = low_color
			pad_plane[start_row:start_row+5, 8:17, 1] = high_color
			start_row = start_row - 5 
			low_color = low_color + 35
			high_color = high_color + 35

		return pad_plane


	def yellow_range(pad_plane, rows):

		start_row = 105
		color = 220

		for i in range(rows):
			pad_plane[start_row:start_row+5, 8:17, 2] = color
			start_row = start_row - 5 
			color = color - 15

		return pad_plane


	def orange_range(pad_plane, rows):

		start_row = 70
		color = 210
		for i in range(rows):
			pad_plane[start_row:start_row+5, 8:17, 1] = color - 15
			pad_plane[start_row:start_row+5, 8:17, 2] = color
			start_row = start_row - 5 
			color = color - 15

		return pad_plane


	def red_range(pad_plane, rows):

		start_row = 35
		color = 250

		for i in range(rows):
			pad_plane[start_row:start_row+5, 8:17, 0] = color
			pad_plane[start_row:start_row+5, 8:17, 1] = 50
			pad_plane[start_row:start_row+5, 8:17, 2] = 50
			start_row = start_row - 5 
			color = color - 15

		return pad_plane


	def tot_energy_to_mev(tot_energy):
		"""Convert tot_energy to energy in MeV using the line of best fit."""
		# Calculate the gradient and y-intercept for the line of best fit
		x1, y1 = 156745, 0.806 # Calibration values
		x2, y2 = 320842, 1.679
		gradient = (y2 - y1) / (x2 - x1)
		y_intercept = y1 - gradient * x1
		return gradient * tot_energy + y_intercept


	def fill_energy_bar(pad_plane, tot_energy):
		"""
		Fills the energy bar where the amount of pixels fired and the color corresponds to the energy of the track
		Max pixel_range should be 28 (7 rows for each color), so need to adjust accordingly.
		"""
		# Calculate the energy in MeV
		energy_mev = tot_energy_to_mev(tot_energy)

		# Calculate the proportion of the energy bar that should be filled
		proportion_filled = energy_mev / 3

		# Calculate how many rows should be filled
		total_rows = math.floor(proportion_filled * 28)

		# Fill the energy bar one row at a time
		if total_rows > 0:
			pad_plane = blue_range(pad_plane, rows=min(total_rows, 7))
		if total_rows > 7:
			pad_plane = yellow_range(pad_plane, rows=min(total_rows-7, 7))
		if total_rows > 14:
			pad_plane = orange_range(pad_plane, rows=min(total_rows-14, 7))
		if total_rows > 21:
			pad_plane = red_range(pad_plane, rows=min(total_rows-21, 7))

		return pad_plane


	def pos_odd_even(event_value):
		"""
		Makes correction to positive points if they are odd or even
		"""
		if event_value % 2 == 0:
			event_value = event_value + 1
			return event_value

		else:
			return event_value


	def neg_odd_even(event_value):
		"""
		Makes correction to negative points if they are odd or even
		"""
		if event_value % 2 == 0:
			event_value = event_value - 1
			return event_value

		else:
			return event_value


	def pt_shift(xset, yset):
		"""
		Shifts all points to the center of nearest pad for pad mapping
		"""
		for j in range(len(xset)):

			if xset[j] > 0:
				xset[j] = math.floor(xset[j])
				pos_adj_valx = pos_odd_even(xset[j])
				xset[j] = pos_adj_valx

			elif xset[j] < 0:
				xset[j] = math.ceil(xset[j])
				neg_adj_valx = neg_odd_even(xset[j])
				xset[j] = neg_adj_valx

			if yset[j] > 0:
				yset[j] = math.floor(yset[j])
				pos_adj_valy = pos_odd_even(yset[j])
				yset[j] = pos_adj_valy

			elif yset[j] < 0:
				yset[j] = math.ceil(yset[j])
				neg_adj_valy = neg_odd_even(yset[j])
				yset[j] = neg_adj_valy

		return xset, yset


	def make_box(mm_grid):
		"""
		Draws the box for the energy bar
		"""
		box_row = np.array([4, 4])
		to_box_row = np.array([145, 146])
		for_box_col = np.array([7, 17])

		box_col = np.array([7, 7])
		to_box_col = np.array([17, 17])
		for_box_row = np.array([4, 145])

		# Draw vertical lines of energy bar box
		for i in range(len(box_row)):
			mm_grid[box_row[i]:to_box_row[i], for_box_col[i], :] = 0
			mm_grid[for_box_row[i], box_col[i]:to_box_col[i], :] = 0

		return mm_grid


	def fill_padplane(xset, yset, eset, tot_energy):
		"""
		Fills the 2D pad plane grid for image creation
		"""
		pad_plane = make_grid()

		xset = np.array(xset)
		yset = np.array(yset)
		eset = np.array(eset)

		# pad plane mapping
		x = (35 + xset) * 2 + 42    # col value
		y = 145 - (35 + yset) * 2   # row value

		# create a dictionary to store (x,y) as keys and e as values
		d = {}
		for i in range(len(x)):
			key = (x[i], y[i])
			if key in d:
				d[key] += eset[i]
			else:
				d[key] = eset[i]

		# convert the dictionary back to arrays
		x = np.zeros(len(d))
		y = np.zeros(len(d))
		eset = np.zeros(len(d))
		for i, key in enumerate(d):
			x[i] = key[0]
			y[i] = key[1]
			eset[i] = d[key]
		    

		# Find max E value and normalize
		energy = eset
		max_energy = np.max(energy)
		norm_energy = energy / max_energy


		# Fill in pad plane   
		for k in range(len(x)):
		
			if y[k] < 9:
				y[k] = y[k] + 4

			if x[k] < 50:
				x[k] = x[k] + 4

			if x[k] > 174:
				x[k] = x[k] - 4

			if y[k] > 53:
				y[k] = y[k] - 4

			if x[k] > 134:
				x[k] = x[k] - 4

			if y[k] > 93:
				y[k] = y[k] - 4

			if y[k] > 133:
				y[k] = y[k] - 4	

			if x[k] < 90:
				x[k] = x[k] + 4


			pad_plane[int(y[k])-1:int(y[k])+2, int(x[k])-1:int(x[k])+2, 0] = norm_energy[k] * 205

			pad_plane[int(y[k])-1:int(y[k])+2, int(x[k])-1:int(x[k])+2, 1] = norm_energy[k] * 240
		
		pad_plane = fill_energy_bar(pad_plane, tot_energy)

		return pad_plane

	def trace_image(padplane_image, trace):
		"""
		Creates a 2D image from trace data
		"""
		# Save plot as jpeg (only want RGB channels, not an alpha channel)
		# Need to take monitor dpi into account to get correct pixel size
		# Plot should have a pixel size of 73x224

		my_dpi = 96
		fig, ax = plt.subplots(figsize=(224/my_dpi, 73/my_dpi))

		x = np.linspace(0, len(trace)-1, len(trace))
		
		ax.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)
		ax.spines['bottom'].set_visible(False)
		ax.spines['left'].set_visible(False)
		ax.fill_between(x, trace, color='b', alpha=1)
		rand_num = random.randrange(0,1000000,1)
		temp_strg = f'/mnt/projects/e21072/OfflineAnalysis/analysis_scripts/energy_depo_{rand_num}.jpg'
		plt.savefig(temp_strg, dpi=my_dpi)
		plt.close()

		# Load png plot as a matrix so that it can be appended to pad plane plot
		img = plt.imread(temp_strg)
		os.remove(temp_strg)
		rows,cols,colors = img.shape # gives dimensions for RGB array
		img_size = rows*cols*colors
		img_1D_vector = img.reshape(img_size)

		# you can recover the orginal image with:
		trace_image = img_1D_vector.reshape(rows,cols,colors)

		# append pad plane image with trace image
		complete_image = np.append(padplane_image, trace_image, axis=0)

		return complete_image
	

	def plot_track():
		orig_num = int(entry_event_num.get())
		event_num = np.where(good_events == orig_num)[0][0]
		#print('EVENT NUM', event_num)
		xHit_list = np.load(os.path.join(sub_mymainpath, 'xHit_list.npy'), allow_pickle=True)
		yHit_list = np.load(os.path.join(sub_mymainpath, 'yHit_list.npy'), allow_pickle=True)
		eHit_list = np.load(os.path.join(sub_mymainpath, 'eHit_list.npy'), allow_pickle=True)

		xHit = xHit_list[event_num]
		yHit = yHit_list[event_num]
		eHit = eHit_list[event_num]

		trace = trace_list[event_num]

		energy = tot_energy[event_num]

		mm_grid = make_grid()
		pad_plane = np.repeat(mm_grid[np.newaxis, :, :], 1, axis=0)
		new_pad_plane = np.repeat(mm_grid[np.newaxis, :, :], 1, axis=0)

		    
		# Call pt_shift function to move all 2D pts to pad centers
		dset_0_copyx, dset_0_copyy = pt_shift(xHit, yHit)
			
		# Call fill_padplane to create 2D pad plane image
		pad_plane = np.append(pad_plane, new_pad_plane, axis=0)
		pad_plane[0] = fill_padplane(dset_0_copyx, dset_0_copyy, eHit, energy)

		# Call trace_image() to append trace to pad plane image
		complete_image = (trace_image(pad_plane[0], trace))

		title = "Particle Track"
		plt.rcParams['figure.figsize'] = [7, 7]
		plt.title(f' Image {orig_num} of {title} Event:', fontdict = {'fontsize' : 20})
		plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
		plt.imshow(complete_image)
		plt.show()

	def plot_track_proj():
		#from scipy.stats import binned_statistic
		from scipy.stats import gaussian_kde
		orig_num = int(entry_event_num.get())
		event_num = np.where(good_events == orig_num)[0][0]
		nbins = 10
		extend_bins = 10

		#print('EVENT NUM', event_num)
		xHit_list = np.load(os.path.join(sub_mymainpath, 'xHit_list.npy'), allow_pickle=True)
		yHit_list = np.load(os.path.join(sub_mymainpath, 'yHit_list.npy'), allow_pickle=True)
		zHit_list = np.load(os.path.join(sub_mymainpath, 'zHit_list.npy'), allow_pickle=True)
		eHit_list = np.load(os.path.join(sub_mymainpath, 'eHit_list.npy'), allow_pickle=True)

		xHit = xHit_list[event_num]
		yHit = yHit_list[event_num]
		zHit = zHit_list[event_num]
		eHit = eHit_list[event_num]

		#trace = trace_list[event_num]

		# fit a line through the point cloud
		A = np.vstack([xHit, yHit, np.ones(len(xHit))]).T
		m, c, _ = np.linalg.lstsq(A, zHit, rcond=None)[0]

		# project the points onto the line
		xProj = (xHit + yHit * m - m * c) / (1 + m**2)
		yProj = (m * xHit + (m**2) * yHit + c * m) / (1 + m**2)
		zProj = m * xProj + c

		# calculate the distance along the line for each point
		dist = np.sqrt((xProj - xHit)**2 + (yProj - yHit)**2)

		kde = gaussian_kde(dist, weights=eHit)
		# division_factor = 3
		division_factor = float(entry_bandwidth.get())
		kde.set_bandwidth(kde.factor / division_factor)  # You can adjust the bandwidth to control the smoothness

		# Create a dense array of x values for the histogram
		x_dense = np.linspace(np.min(dist) - extend_bins, np.max(dist) + extend_bins, 1000)

		# Evaluate the KDE for the dense x values
		y_smooth = kde.evaluate(x_dense)

		# Plot the smooth histogram
		fig, ax = plt.subplots()
		ax.plot(x_dense, y_smooth)
		ax.set_xlabel('Energy Bins')
		ax.set_ylabel('Probability Density')
		ax.set_title('Probability Density Function of Energy\nDistribution after Projection')
		plt.show()

		return x_dense, y_smooth

	def fit_trace():
		from scipy.stats import gaussian_kde
		orig_num = int(entry_event_num.get())
		event_num = np.where(good_events == orig_num)[0][0]
		nbins = 10
		extend_bins = 10

		#print('EVENT NUM', event_num)
		xHit_list = np.load(os.path.join(sub_mymainpath, 'xHit_list.npy'), allow_pickle=True)
		yHit_list = np.load(os.path.join(sub_mymainpath, 'yHit_list.npy'), allow_pickle=True)
		zHit_list = np.load(os.path.join(sub_mymainpath, 'zHit_list.npy'), allow_pickle=True)
		eHit_list = np.load(os.path.join(sub_mymainpath, 'eHit_list.npy'), allow_pickle=True)

		xHit = xHit_list[event_num]
		yHit = yHit_list[event_num]
		zHit = zHit_list[event_num]
		eHit = eHit_list[event_num]

		#trace = trace_list[event_num]

		# fit a line through the point cloud
		A = np.vstack([xHit, yHit, np.ones(len(xHit))]).T
		m, c, _ = np.linalg.lstsq(A, zHit, rcond=None)[0]

		# project the points onto the line
		xProj = (xHit + yHit * m - m * c) / (1 + m**2)
		yProj = (m * xHit + (m**2) * yHit + c * m) / (1 + m**2)
		zProj = m * xProj + c

		# calculate the distance along the line for each point
		dist = np.sqrt((xProj - xHit)**2 + (yProj - yHit)**2)

		kde = gaussian_kde(dist, weights=eHit)
		# division_factor = 3
		division_factor = float(entry_bandwidth.get())
		kde.set_bandwidth(kde.factor / division_factor)  # You can adjust the bandwidth to control the smoothness

		# Create a dense array of x values for the histogram
		x_dense = np.linspace(np.min(dist) - extend_bins, np.max(dist) + extend_bins, 1000)

		# Evaluate the KDE for the dense x values
		y_smooth = kde.evaluate(x_dense)
	
		plot_smooth_histogram_fit(x_dense, y_smooth, division_factor)

		return
		
	


	def plot_smooth_histogram_fit(x_dense, y_smooth, division_factor):
		from scipy.special import erfc
		from scipy.optimize import curve_fit
		from scipy.stats import chisquare, chi2
		from matplotlib.offsetbox import AnchoredOffsetbox, TextArea
	
		global peak_active, peak_handle, peak_info, horizontal_line

		# Extract calibrated energy of point
		calib_point_1 = (0.806, 156745)
		calib_point_2 = (1.679, 320842)
		#calib_point_1 = (0.806, 157600)
		#calib_point_2 = (1.679, 275300)
		#calib_point_1 = (0.303, 84672)
		#calib_point_2 = (2.150, 374439)
		#calib_point_1 = (0.806, 157700)
		#calib_point_2 = (1.679, 308200)

		# Convert calibration points to MeV
		energy_1, channel_1 = calib_point_1
		energy_2, channel_2 = calib_point_2
		energy_scale_factor = (energy_2 - energy_1) / (channel_2 - channel_1)
		energy_offset = energy_1 - energy_scale_factor * channel_1

		# Convert tot_energy to MeV
		tot_energy_MeV = np.array(tot_energy) * energy_scale_factor + energy_offset

		orig_num = int(entry_event_num.get())
		event_num = np.where(good_events == orig_num)[0][0]
		event_energy = tot_energy_MeV[event_num]
		
		def safe_exp(x):
			max_exp_arg = np.finfo(np.float64).maxexp - 10
			return np.exp(np.clip(x, None, max_exp_arg))

		def emg(x, amplitude, mu, sigma, lambda_):
			exp_arg = 0.5 * lambda_ * (2 * mu + lambda_ * sigma**2 - 2 * x)
			exp_term = safe_exp(exp_arg)
			return 0.5 * amplitude * lambda_ * exp_term * erfc((mu + lambda_ * sigma**2 - x) / (np.sqrt(2) * sigma))

		def composite_emg(x, *params):
			result = np.zeros_like(x)
			for i in range(0, len(params), 4):
				result += emg(x, *params[i:i + 4])
			return result

		# Plot the smooth histogram
		fig, ax = plt.subplots()
		ax.plot(x_dense, y_smooth, label='Smooth Histogram')
		peak_handle, = plt.plot([], [], 'o', color='black', markersize=10, alpha=0.7)

		# Initialize peak detection variables
		peak_active = None
		peak_info = []
		horizontal_line = None

		def onclick(event):
			global peak_active, peak_handle, peak_info, horizontal_line
			if event.button == 1:  # Left mouse button
				x, y = event.xdata, event.ydata
				plt.plot(x, y, 'ro', markersize=10)
				plt.axvline(x, color='r', linestyle='--')
				plt.draw()
				peak_active = x

			elif event.button == 3:  # Right mouse button
				if peak_active is not None:
					x, y = event.xdata, event.ydata
					plt.plot(x, y, 'go', markersize=10)
					plt.draw()

					idx = np.argmin(np.abs(x_dense - peak_active))
					amp = y_smooth[idx]
					mu = peak_active
					sigma = np.abs(x - peak_active)
					peak_info.extend([amp, mu, sigma, 1])

					horizontal_line, = plt.plot([peak_active, x], [y, y], color='green', linestyle='--')

					peak_active = None
					plt.draw()

		# Connect the click event to the plot
		cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)

		title1 = "Left Click to Select Peak Amp and Mu"
		title2 = "\nRight Click to Select Peak Sigma"

		# Calculate the position for each part of the title
		x1, y1 = 0.5, 1.10
		x2, y2 = 0.5, 1.05

		# Set the title using ax.annotate() and the ax.transAxes transform
		ax.annotate(title1, (x1, y1), xycoords='axes fraction', fontsize=12, color='red', ha='center', va='center')
		ax.annotate(title2, (x2, y2), xycoords='axes fraction', fontsize=12, color='green', ha='center', va='center')

		plt.show()

		# Fit the composite EMG function to the data
		popt, pcov = curve_fit(composite_emg, x_dense, y_smooth, p0=peak_info, maxfev=1000000)
		fitted_emg = composite_emg(x_dense, *popt)

		# Print final fit parameters
		def display_fit_parameters(peak_info, popt, fixed_list=None):
			fit_params_window = Toplevel()
			fit_params_window.title("Final Fit Parameters")
			fit_params_window.geometry("700x200")
			output_text = Text(fit_params_window, wrap=WORD)
			output_text.pack(expand=True, fill=BOTH)

			param_names = ['Amp', 'Mu', 'Sigma', 'Lambda']
			idx = 0
			fixed_param_idx = 0

			# If fixed_list is not provided, create a list of all False values
			if fixed_list is None:
				fixed_list = [False] * len(peak_info)

			for i in range(0, len(peak_info), 4):
				peak_label = f"Peak {(i // 4) + 1}: "
				for j in range(4):
					if fixed_list[i + j]:
						peak_label += f"*{param_names[j]}={fixed_params[fixed_param_idx]}, "
						fixed_param_idx += 1
					else:
						peak_label += f"{param_names[j]}={popt[idx]}, "
						idx += 1
				output_text.insert(END, peak_label + "\n")


		display_fit_parameters(peak_info, popt)


		# Create a 2x1 grid of subplots
		fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
		fig.subplots_adjust(hspace=0.1)

		# Plot the smooth histogram and the fitted EMG on the main plot (ax1)
		ax1.plot(x_dense, y_smooth, label='Smooth Histogram')
		ax1.fill_between(x_dense, y_smooth, color='#1f77b4')
		ax1.plot(x_dense, fitted_emg, linestyle='--', linewidth=2, label='Composite Fit')
		ax1.set_xlabel('Energy Bins')
		ax1.set_ylabel('Probability Density')

		n_peaks = len(popt) // 4
		# Check if there are 3 peaks
		if n_peaks > 2:
			# The first peak is always separate
			separate_peak_index = 0

			# All other peaks are combined
			same_particle_peak_indices = list(range(1, n_peaks))

			# Calculate the combined EMG curve for the peaks that represent the same particle
			combined_emg_curve = np.zeros_like(x_dense)
			for idx in same_particle_peak_indices:
				combined_emg_curve += emg(x_dense, *popt[idx * 4:idx * 4 + 4])

			# Plot the separate peak
			separate_peak = emg(x_dense, *popt[separate_peak_index * 4:separate_peak_index * 4 + 4])
			ax1.plot(x_dense, separate_peak, linestyle=':', label=f'Peak {separate_peak_index + 1}')

			# Plot the combined curve
			ax1.plot(x_dense, combined_emg_curve, linestyle=':', label='Peak 2 (summed)')

			# Calculate the area under the separate peak and combined peaks
			separate_peak_area = np.trapz(separate_peak, x_dense)
			combined_area = np.trapz(combined_emg_curve, x_dense)

			# Create a list of peak areas
			peak_areas = [separate_peak_area, combined_area]

			# Normalize the sum of peak areas to the event_energy (MeV)
			peak_energy_ratios = [area / sum(peak_areas) for area in peak_areas]
			peak_energies = [event_energy * ratio for ratio in peak_energy_ratios]

		else: 
			# Plot each individual peak
			for i in range(0, len(popt), 4):
				individual_peak = emg(x_dense, *popt[i:i + 4])
				ax1.plot(x_dense, individual_peak, linestyle=':', label=f'Peak {i // 4 + 1}')

			# Calculate the area under each individual peak
			peak_areas = [np.trapz(emg(x_dense, *popt[i:i + 4]), x_dense) for i in range(0, len(popt), 4)]

			# Normalize the sum of peak areas to the event_energy (MeV)
			peak_energy_ratios = [area / sum(peak_areas) for area in peak_areas]
			peak_energies = [event_energy * ratio for ratio in peak_energy_ratios]

		# Calculate residuals
		residuals = y_smooth - fitted_emg

		# Plot the residuals on the residual plot (ax2)
		ax2.plot(x_dense, residuals, linestyle='-', marker='o', markersize=4, label='Residuals')
		ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
		ax2.set_ylabel('Residuals')
	

		# Create a title string with the energy for each peak
		title_string = f"EVENT {orig_num}\nPDF of Energy Distribution (BW: {division_factor}) | Total Energy: {event_energy:.3f} MeV\n"
		for i, peak_energy in enumerate(peak_energies):
			title_string += f"Peak {i + 1}: {peak_energy:.3f} MeV \n"

		# Remove the last comma and space from the title string
		title_string = title_string[:-2]

		# Set the title of the plot
		ax1.set_title(title_string, fontdict={'fontsize': 13})

		# Calculate R-squared
		ss_res = np.sum((y_smooth - fitted_emg)**2)
		ss_tot = np.sum((y_smooth - np.mean(y_smooth))**2)
		r_squared = 1 - (ss_res / ss_tot)
		text_r_squared = f"R-squared: {r_squared:.4f}"

		# Calculate chi-squared and p-value
		observed = y_smooth
		expected = fitted_emg * np.sum(observed) / np.sum(fitted_emg)  # Normalize the expected values
		chi2_stat, p_value = chisquare(observed, expected)
		#degrees_of_freedom = len(observed) - len(popt)
		#chi2_per_dof = chi2_stat / degrees_of_freedom

		# Create a formatted text string
		text_chi = f"Chi-squared: {chi2_stat:.4f}"
		#text_chi_dof = f"Chi-squared/dof: {chi2_per_dof:.4f}"
		text_p_val = f"P-value: {p_value:.4e}"
		print("\n*****Fit Parameters*****")
		for i in range(0, len(popt), 4):
			print(f"Peak {i // 4 + 1}: Amp={popt[i]:.3f}, Mu={popt[i + 1]:.3f}, Sigma={popt[i + 2]:.3f}, Lambda={popt[i + 3]:.3f}\n")

		# Create empty plot with blank marker containing the extra label
		#ax1.plot([], [], ' ', label=text_chi)
		#ax1.plot([], [], ' ', label=text_chi_dof)
		ax1.plot([], [], ' ', label=text_p_val)
		ax1.plot([], [], ' ', label=text_r_squared)
		ax1.legend()


		# Show the plot
		plt.show()

		###### next plot



	def parse_peak_info_string(peak_info_str):
		param_pattern = r"(\*?[A-Za-z]+)=([-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?)"
		param_regex = re.compile(param_pattern)

		peak_lines = peak_info_str.split("\n")

		float_list = []
		fixed_list = []
		for line in peak_lines:
			params = param_regex.findall(line)

			for param, value in params:
				float_list.append(float(value.replace(",", "")))
				fixed_list.append(param.startswith("*"))

		return float_list, fixed_list


	def plot_spectrum_multi_params():
		from scipy.special import erfc
		from scipy.special import erfcx
		from scipy.optimize import curve_fit
		from scipy.stats import chisquare, chi2
		from matplotlib.offsetbox import AnchoredOffsetbox, TextArea
		from scipy.stats import gaussian_kde


		def safe_exp(x, min_exp_arg=None, max_exp_arg=None):
			min_exp_arg = min_exp_arg if min_exp_arg is not None else -np.inf
			max_exp_arg = max_exp_arg if max_exp_arg is not None else np.finfo(np.float64).maxexp - 10
			return np.exp(np.clip(x, min_exp_arg, max_exp_arg))

		
		def emg_stable(x, amplitude, mu, sigma, lambda_):
			exp_arg = 0.5 * lambda_ * (2 * mu + lambda_ * sigma**2 - 2 * x)
			erfc_arg = (mu + lambda_ * sigma**2 - x) / (np.sqrt(2) * sigma)
			#print("lambda_: ", lambda_)
			#print("mu: ", mu)
			#print("sigma: ", sigma)
			#print("x: ", x)
			return 0.5 * amplitude * lambda_ * safe_exp(exp_arg - erfc_arg**2) * erfcx(erfc_arg)

		def safe_exp(x):
			max_exp_arg = np.finfo(np.float64).maxexp - 10
			return np.exp(np.clip(x, None, max_exp_arg))

		def emg(x, amplitude, mu, sigma, lambda_):
			exp_arg = 0.5 * lambda_ * (2 * mu + lambda_ * sigma**2 - 2 * x)
			exp_term = safe_exp(exp_arg)
			return 0.5 * amplitude * lambda_ * exp_term * erfc((mu + lambda_ * sigma**2 - x) / (np.sqrt(2) * sigma))


		def composite_emg(x, params, fixed_params, fixed_list):
			result = np.zeros_like(x)
			fixed_param_idx = 0
			param_idx = 0
			for i in range(0, len(params) + len(fixed_params), 4):
				current_params = [0] * 4
				for j in range(4):
					if fixed_list[i + j]:
						current_params[j] = fixed_params[fixed_param_idx]
						fixed_param_idx += 1
					else:
						current_params[j] = params[param_idx]
						param_idx += 1
				result += emg_stable(x, *current_params)
			return result



		orig_num = int(entry_event_num.get())
		event_num = np.where(good_events == orig_num)[0][0]
		nbins = 10
		extend_bins = 10

		#print('EVENT NUM', event_num)
		xHit_list = np.load(os.path.join(sub_mymainpath, 'xHit_list.npy'), allow_pickle=True)
		yHit_list = np.load(os.path.join(sub_mymainpath, 'yHit_list.npy'), allow_pickle=True)
		zHit_list = np.load(os.path.join(sub_mymainpath, 'zHit_list.npy'), allow_pickle=True)
		eHit_list = np.load(os.path.join(sub_mymainpath, 'eHit_list.npy'), allow_pickle=True)

		xHit = xHit_list[event_num]
		yHit = yHit_list[event_num]
		zHit = zHit_list[event_num]
		eHit = eHit_list[event_num]

		#trace = trace_list[event_num]

		# fit a line through the point cloud
		A = np.vstack([xHit, yHit, np.ones(len(xHit))]).T
		m, c, _ = np.linalg.lstsq(A, zHit, rcond=None)[0]

		# project the points onto the line
		xProj = (xHit + yHit * m - m * c) / (1 + m**2)
		yProj = (m * xHit + (m**2) * yHit + c * m) / (1 + m**2)
		zProj = m * xProj + c

		# calculate the distance along the line for each point
		dist = np.sqrt((xProj - xHit)**2 + (yProj - yHit)**2)

		kde = gaussian_kde(dist, weights=eHit)
		# division_factor = 3
		division_factor = float(entry_bandwidth.get())
		kde.set_bandwidth(kde.factor / division_factor)  # You can adjust the bandwidth to control the smoothness

		# Create a dense array of x values for the histogram
		x_dense = np.linspace(np.min(dist) - extend_bins, np.max(dist) + extend_bins, 1000)

		# Evaluate the KDE for the dense x values
		y_smooth = kde.evaluate(x_dense)

		# Get Peak info
		input_str = entry_multi.get()
		peak_info, fixed_list = parse_peak_info_string(input_str)

		# Filter out fixed parameters from the initial guess for the curve_fit function
		initial_guess = [peak_info[i] for i in range(len(peak_info)) if not fixed_list[i]]

		# Fit the composite EMG function to the data
		fixed_params = [peak_info[i] for i in range(len(peak_info)) if fixed_list[i]]
		popt, pcov = curve_fit(lambda x, *params: composite_emg(x, params, fixed_params, fixed_list), x_dense, y_smooth, p0=initial_guess, maxfev=1000000)
		fitted_emg = composite_emg(x_dense, popt, fixed_params, fixed_list)


		# Print final fit parameters
		print("FINAL FIT PARAMETERS:")
		param_names = ['Amp', 'Mu', 'Sigma', 'Lambda']
		idx = 0
		fixed_param_idx = 0

		for i in range(0, len(peak_info), 4):
			print(f"Peak {(i // 4) + 1}:", end=" ")
			for j in range(4):
				if fixed_list[i + j]:
					print(f"*{param_names[j]}={fixed_params[fixed_param_idx]}", end=", ")
					fixed_param_idx += 1
				else:
					print(f"{param_names[j]}={popt[idx]}", end=", ")
					idx += 1
			print()

		# Print final fit parameters
		def display_fit_parameters(peak_info, popt, fixed_list=None):
			fit_params_window = Toplevel()
			fit_params_window.title("Final Fit Parameters")
			fit_params_window.geometry("700x200")
			output_text = Text(fit_params_window, wrap=WORD)
			output_text.pack(expand=True, fill=BOTH)

			param_names = ['Amp', 'Mu', 'Sigma', 'Lambda']
			idx = 0
			fixed_param_idx = 0

			# If fixed_list is not provided, create a list of all False values
			if fixed_list is None:
				fixed_list = [False] * len(peak_info)

			for i in range(0, len(peak_info), 4):
				peak_label = f"Peak {(i // 4) + 1}: "
				for j in range(4):
					if fixed_list[i + j]:
						peak_label += f"*{param_names[j]}={fixed_params[fixed_param_idx]}, "
						fixed_param_idx += 1
					else:
						peak_label += f"{param_names[j]}={popt[idx]}, "
						idx += 1
				output_text.insert(END, peak_label + "\n")


		display_fit_parameters(peak_info, popt, fixed_list)
	
		"""
		# Convert mu integrated charge to energy
		mu_values = [mu * energy_scale_factor + energy_offset for mu in popt[1::4]]

		# Add peak labels
		for idx, mu_value in enumerate(mu_values):
			y_value = fitted_emg[np.argmin(np.abs(filtered_x_hist_MeV - mu_value))]
			plt.annotate(f"< {mu_value:.2f} MeV | Peak {idx+1}", (mu_value, y_value),textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, color='black', rotation=90)

		plt.ylim(0, 1.55 * np.max(fitted_emg))
		plt.show()
		"""

		# Create a 2x1 grid of subplots
		fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
		fig.subplots_adjust(hspace=0.1)

		# Plot the smooth histogram and the fitted EMG on the main plot (ax1)
		ax1.plot(x_dense, y_smooth, label='Smooth Histogram')
		ax1.fill_between(x_dense, y_smooth, color='#1f77b4')
		ax1.plot(x_dense, fitted_emg, linestyle='--', label='Fitted EMG')
		ax1.set_xlabel('Energy Bins')
		ax1.set_ylabel('Probability Density')
	

		# Calculate residuals
		residuals = y_smooth - fitted_emg

		# Plot the residuals on the residual plot (ax2)
		ax2.plot(x_dense, residuals, linestyle='-', marker='o', markersize=4, label='Residuals')
		ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
		ax2.set_ylabel('Residuals')

		def peak_areas(peak_info, popt, fixed_list=None):
			areas = []
			idx = 0
			fixed_param_idx = 0

			# If fixed_list is not provided, create a list of all False values
			if fixed_list is None:
				fixed_list = [False] * len(peak_info)

			for i in range(0, len(peak_info), 4):
				if not fixed_list[i]:  # Check if Amp is not fixed
					amp = popt[idx]
					idx += 1
				else:
					amp = fixed_params[fixed_param_idx]
					fixed_param_idx += 1

				if not fixed_list[i + 2]:  # Check if Sigma is not fixed
					sigma = popt[idx + 1]
					idx += 1
				else:
					sigma = fixed_params[fixed_param_idx]
					fixed_param_idx += 1

				area = amp * sigma * math.sqrt(2 * math.pi)
				areas.append(area)
			return areas


		peak_areas = peak_areas(peak_info, popt, fixed_list)		

		# Extract calibrated energy of point
		calib_point_1 = (0.806, 156745)
		calib_point_2 = (1.679, 320842)
		#calib_point_1 = (0.806, 157600)
		#calib_point_2 = (1.679, 275300)
		#calib_point_1 = (0.303, 84672)
		#calib_point_2 = (2.150, 374439)
		#calib_point_1 = (0.806, 157700)
		#calib_point_2 = (1.679, 308200)

		# Convert calibration points to MeV
		energy_1, channel_1 = calib_point_1
		energy_2, channel_2 = calib_point_2
		energy_scale_factor = (energy_2 - energy_1) / (channel_2 - channel_1)
		energy_offset = energy_1 - energy_scale_factor * channel_1

		# Convert tot_energy to MeV
		tot_energy_MeV = np.array(tot_energy) * energy_scale_factor + energy_offset

		orig_num = int(entry_event_num.get())
		event_num = np.where(good_events == orig_num)[0][0]
		event_energy = tot_energy_MeV[event_num]


		# Normalize the sum of peak areas to the event_energy (MeV)
		peak_energy_ratios = [area / sum(peak_areas) for area in peak_areas]
		peak_energies = [event_energy * ratio for ratio in peak_energy_ratios]

		# Create a title string with the energy for each peak
		title_string = f"EVENT {orig_num}\nPDF of Energy Distribution (BW: {division_factor}) | Total Energy: {event_energy:.3f} MeV\n"
		for i, peak_energy in enumerate(peak_energies):
			title_string += f"Peak {i + 1}: {peak_energy:.3f} MeV \n"

		# Remove the last comma and space from the title string
		title_string = title_string[:-2]

		# Set the title of the plot
		ax1.set_title(title_string, fontdict={'fontsize': 10})

		# Calculate R-squared
		ss_res = np.sum((y_smooth - fitted_emg)**2)
		ss_tot = np.sum((y_smooth - np.mean(y_smooth))**2)
		r_squared = 1 - (ss_res / ss_tot)
		text_r_squared = f"R-squared: {r_squared:.4f}"

		# Calculate Mean Squared Error (MSE)
		mse = np.mean((y_smooth - fitted_emg)**2)
		text_mse = f"MSE: {mse:.4f}"

		# Calculate chi-squared and p-value
		observed = y_smooth
		expected = fitted_emg * np.sum(observed) / np.sum(fitted_emg)  # Normalize the expected values
		chi2_stat, p_value = chisquare(observed, expected)
		#degrees_of_freedom = len(observed) - len(popt)
		#chi2_per_dof = chi2_stat / degrees_of_freedom

		# Create a formatted text string
		text_chi = f"Chi-squared: {chi2_stat:.4f}"
		#text_chi_dof = f"Chi-squared/dof: {chi2_per_dof:.4f}"
		text_p_val = f"P-value: {p_value:.4e}"
		"""
		print("\n*****Fit Parameters*****")
		for i in range(0, len(popt), 4):
			print(f"Peak {i // 4 + 1}: Amp={popt[i]:.3f}, Mu={popt[i + 1]:.3f}, Sigma={popt[i + 2]:.3f}, Lambda={popt[i + 3]:.3f}\n")
		"""

		# Create empty plot with blank marker containing the extra label
		ax1.plot([], [], ' ', label=text_chi)
		#ax1.plot([], [], ' ', label=text_chi_dof)
		ax1.plot([], [], ' ', label=text_p_val)
		ax1.plot([], [], ' ', label=text_r_squared)
		ax1.plot([], [], ' ', label=text_mse)
		ax1.legend()


		# Show the plot
		plt.show()

	def fit_init_trace():
		from scipy.special import erfc
		from scipy.special import erfcx
		from scipy.optimize import curve_fit
		from scipy.stats import chisquare, chi2
		from matplotlib.offsetbox import AnchoredOffsetbox, TextArea
		from scipy.integrate import quad
	
		global peak_active, peak_handle, peak_info, horizontal_line

		# Extract calibrated energy of point
		calib_point_1 = (0.806, 156745)
		calib_point_2 = (1.679, 320842)
		#calib_point_1 = (0.806, 157600)
		#calib_point_2 = (1.679, 275300)
		#calib_point_1 = (0.303, 84672)
		#calib_point_2 = (2.150, 374439)
		#calib_point_1 = (0.806, 157700)
		#calib_point_2 = (1.679, 308200)

		# Convert calibration points to MeV
		energy_1, channel_1 = calib_point_1
		energy_2, channel_2 = calib_point_2
		energy_scale_factor = (energy_2 - energy_1) / (channel_2 - channel_1)
		energy_offset = energy_1 - energy_scale_factor * channel_1

		# Convert tot_energy to MeV
		tot_energy_MeV = np.array(tot_energy) * energy_scale_factor + energy_offset

		orig_num = int(entry_event_num.get())
		event_num = np.where(good_events == orig_num)[0][0]
		event_energy = tot_energy_MeV[event_num]
		y_trace = trace_list[event_num]

		fig, ax = plt.subplots()
		x_trace = np.linspace(0, len(y_trace)-1, len(y_trace))
		ax.fill_between(x_trace, y_trace, color='#1f77b4', alpha=1)

		
		def safe_exp(x, min_exp_arg=None, max_exp_arg=None):
			min_exp_arg = min_exp_arg if min_exp_arg is not None else -np.inf
			max_exp_arg = max_exp_arg if max_exp_arg is not None else np.finfo(np.float64).maxexp - 10
			return np.exp(np.clip(x, min_exp_arg, max_exp_arg))

		
		def emg_stable(x, amplitude, mu, sigma, lambda_):
			exp_arg = 0.5 * lambda_ * (2 * mu + lambda_ * sigma**2 - 2 * x)
			erfc_arg = (mu + lambda_ * sigma**2 - x) / (np.sqrt(2) * sigma)
			#print("lambda_: ", lambda_)
			#print("mu: ", mu)
			#print("sigma: ", sigma)
			#print("x: ", x)
			return 0.5 * amplitude * lambda_ * safe_exp(exp_arg - erfc_arg**2) * erfcx(erfc_arg)

		def emg(x, amplitude, mu, sigma, lambda_):
			exp_arg = 0.5 * lambda_ * (2 * mu + lambda_ * sigma**2 - 2 * x)
			exp_term = safe_exp(exp_arg)
			return 0.5 * amplitude * lambda_ * exp_term * erfc((mu + lambda_ * sigma**2 - x) / (np.sqrt(2) * sigma))


		def composite_emg(x, *params):
			result = np.zeros_like(x)
			for i in range(0, len(params), 4):
				result += emg_stable(x, *params[i:i + 4])
			return result
		

	
		peak_handle, = plt.plot([], [], 'o', color='black', markersize=10, alpha=0.7)

		# initialize peak detection variables
		peaks = []
		peak_data = []
		peak_active = None
		peak_info = []

		# keep track of the last left-click point
		last_left_click = None

		def onclick(event):
			global peak_active, peak_handle, peak_info, horizontal_line
			if event.button == 1:  # Left mouse button
				x, y = event.xdata, event.ydata
				plt.plot(x, y, 'ro', markersize=10)
				plt.axvline(x, color='r', linestyle='--')
				plt.draw()
				peak_active = x

			elif event.button == 3:  # Right mouse button
				if peak_active is not None:
					x, y = event.xdata, event.ydata
					plt.plot(x, y, 'go', markersize=10)
					plt.draw()

					idx = np.argmin(np.abs(x_trace - peak_active))
					mu = peak_active
					sigma = np.abs(x - peak_active)
					amp = y_trace[idx] * np.sqrt(2 * np.pi) * sigma
					peak_info.extend([amp, mu, sigma, 1])

					horizontal_line, = plt.plot([peak_active, x], [y, y], color='green', linestyle='--')

					peak_active = None
					plt.draw()

		# Connect the click event to the plot
		cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)

		title1 = "Left Click to Select Peak Amp and Mu"
		title2 = "\nRight Click to Select Peak Sigma"

		# Calculate the position for each part of the title
		x1, y1 = 0.5, 1.10
		x2, y2 = 0.5, 1.05

		# Set the title using ax.annotate() and the ax.transAxes transform
		ax.annotate(title1, (x1, y1), xycoords='axes fraction', fontsize=12, color='red', ha='center', va='center')
		ax.annotate(title2, (x2, y2), xycoords='axes fraction', fontsize=12, color='green', ha='center', va='center')

		plt.show()

		# Fit the composite EMG function to the data
		popt, pcov = curve_fit(composite_emg, x_trace, y_trace, p0=peak_info, maxfev=1000000)
		fitted_emg = composite_emg(x_trace, *popt)

		# Print final fit parameters
		def display_fit_parameters(peak_info, popt, fixed_list=None):
			fit_params_window = Toplevel()
			fit_params_window.title("Final Fit Parameters")
			fit_params_window.geometry("700x200")
			output_text = Text(fit_params_window, wrap=WORD)
			output_text.pack(expand=True, fill=BOTH)

			param_names = ['Amp', 'Mu', 'Sigma', 'Lambda']
			idx = 0
			fixed_param_idx = 0

			# If fixed_list is not provided, create a list of all False values
			if fixed_list is None:
				fixed_list = [False] * len(peak_info)

			for i in range(0, len(peak_info), 4):
				peak_label = f"Peak {(i // 4) + 1}: "
				for j in range(4):
					if fixed_list[i + j]:
						peak_label += f"*{param_names[j]}={fixed_params[fixed_param_idx]}, "
						fixed_param_idx += 1
					else:
						peak_label += f"{param_names[j]}={popt[idx]}, "
						idx += 1
				output_text.insert(END, peak_label + "\n")


		display_fit_parameters(peak_info, popt)


		# Create a 2x1 grid of subplots
		fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
		fig.subplots_adjust(hspace=0.1)

		# Plot the smooth histogram and the fitted EMG on the main plot (ax1)
		#ax1.hist(x_hist, bins=len(trace), label='Raw Trace')
		#ax1.fill_between(x_hist, y_hist, color='#1f77b4')
		ax1.fill_between(x_trace, y_trace, color='#1f77b4', alpha=1)
		ax1.plot(x_trace, fitted_emg, linestyle='--', color='red', linewidth=2, label='Composite Fit')
		ax1.set_xlabel('Energy Bins')
		ax1.set_ylabel('Integrated Charge')

		n_peaks = len(popt) // 4
		# Check if there are 3 peaks
		if n_peaks > 2:
			
			########## Code for summing all peaks after the first peak
			# The first peak is always separate
			separate_peak_index = 0

			# All other peaks are combined
			same_particle_peak_indices = list(range(1, n_peaks))

			# Calculate the combined EMG curve for the peaks that represent the same particle
			combined_emg_curve = np.zeros_like(x_trace)
			for idx in same_particle_peak_indices:
				combined_emg_curve += emg_stable(x_trace, *popt[idx * 4:idx * 4 + 4])

			# Plot the separate peak
			separate_peak = emg_stable(x_trace, *popt[separate_peak_index * 4:separate_peak_index * 4 + 4])
			ax1.plot(x_trace, separate_peak, linestyle=':', color='blue', label=f'Peak {separate_peak_index + 1}')

			# Plot the combined curve
			ax1.plot(x_trace, combined_emg_curve, linestyle=':', color='orange', label='Peak 2 (summed)')

			# Calculate the area under the separate peak and combined peaks
			separate_peak_area = np.trapz(separate_peak, x_trace)
			combined_area = np.trapz(combined_emg_curve, x_trace)

			# Create a list of peak areas
			peak_areas = [separate_peak_area, combined_area]

			# Normalize the sum of peak areas to the event_energy (MeV)
			peak_energy_ratios = [area / sum(peak_areas) for area in peak_areas]
			peak_energies = [event_energy * ratio for ratio in peak_energy_ratios]
			#####################################################################
			"""
			########## Code for summing different peak combos 
			# Specify the groups of peaks that should be combined
			combined_peak_groups = [[0, 1], [2, 3]]

			# Initialize a list for the combined EMG curves and peak areas
			combined_emg_curves = []
			combined_areas = []

			# Calculate the combined EMG curves and areas for each group
			for group in combined_peak_groups:
				combined_emg_curve = np.zeros_like(x_trace)
				for idx in group:
					combined_emg_curve += emg_stable(x_trace, *popt[idx * 4:idx * 4 + 4])
				combined_emg_curves.append(combined_emg_curve)
				combined_areas.append(np.trapz(combined_emg_curve, x_trace))

			# Plot the individual peaks that are not part of any group
			for i in range(0, len(popt), 4):
				if i // 4 not in [idx for group in combined_peak_groups for idx in group]:
					individual_peak = emg_stable(x_trace, *popt[i:i + 4])
					ax1.plot(x_trace, individual_peak, linestyle=':', color='green', label=f'Peak {i // 4 + 1}')

			# Plot the combined curves
			colors = ['blue', 'orange']
			for i, combined_emg_curve in enumerate(combined_emg_curves):
				ax1.plot(x_trace, combined_emg_curve, linestyle=':', color=colors[i], label=f'Combined Peaks {i + 1}')

			# Calculate the area under each individual peak that is not part of any group
			peak_areas = [np.trapz(emg_stable(x_trace, *popt[i:i + 4]), x_trace) for i in range(0, len(popt), 4) if i // 4 not in [idx for group in combined_peak_groups for idx in group]]

			# Add the combined areas to the list of peak areas
			peak_areas.extend(combined_areas)

			# Normalize the sum of peak areas to the event_energy (MeV)
			peak_energy_ratios = [area / sum(peak_areas) for area in peak_areas]
			peak_energies = [event_energy * ratio for ratio in peak_energy_ratios]
			#############################################
			"""

		else: 
			# Plot each individual peak
			colors = ['blue', 'orange']
			for i in range(0, len(popt), 4):
				individual_peak = emg_stable(x_trace, *popt[i:i + 4])
				color_idx = (i // 4) % len(colors)  # Cycle through the colors
				ax1.plot(x_trace, individual_peak, linestyle=':', color=colors[color_idx], label=f'Peak {i // 4 + 1}')

			# Calculate the area under each individual peak
			peak_areas = [np.trapz(emg_stable(x_trace, *popt[i:i + 4]), x_trace) for i in range(0, len(popt), 4)]

			# Normalize the sum of peak areas to the event_energy (MeV)
			peak_energy_ratios = [area / sum(peak_areas) for area in peak_areas]
			peak_energies = [event_energy * ratio for ratio in peak_energy_ratios]
	

		# Calculate residuals
		residuals = y_trace - fitted_emg

		# Plot the residuals on the residual plot (ax2)
		ax2.plot(x_trace, residuals, linestyle='-', marker='o', markersize=4, label='Residuals')
		ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
		ax2.set_ylabel('Residuals')
	

		# Create a title string with the energy for each peak
		title_string = f"EVENT {orig_num}\nFit of Trace | Total Energy: {event_energy:.3f} MeV\n"
		for i, peak_energy in enumerate(peak_energies):
			title_string += f"Peak {i + 1}: {peak_energy:.3f} MeV \n"

		# Remove the last comma and space from the title string
		title_string = title_string[:-2]

		# Set the title of the plot
		ax1.set_title(title_string, fontdict={'fontsize': 13})

		# Calculate R-squared
		ss_res = np.sum((y_trace - fitted_emg)**2)
		ss_tot = np.sum((y_trace - np.mean(y_trace))**2)
		r_squared = 1 - (ss_res / ss_tot)
		text_r_squared = f"R-squared: {r_squared:.4f}"

		# Calculate Mean Squared Error (MSE)
		mse = np.mean((y_trace - fitted_emg)**2)
		text_mse = f"MSE: {mse:.4f}"

		# Create empty plot with blank marker containing the extra label
		#ax1.plot([], [], ' ', label=text_chi)
		#ax1.plot([], [], ' ', label=text_chi_dof)
		ax1.plot([], [], ' ', label=text_r_squared)
		ax1.plot([], [], ' ', label=text_mse)
		ax1.legend()

		# Show the plot
		plt.show()



	def on_focus_in(event):
		if event.widget.get() == event.widget.default_text:
			event.widget.delete(0, END)

	def on_focus_out(event):
		if event.widget.get() == '':
			event.widget.insert(0, event.widget.default_text)


	def temp_text7(e):
		entry_event_num.delete(0,"end")

	def temp_text61(e):
		entry_multi.delete(0,"end")


	def close_out():
		entry_event_num.destroy()
		button_plot.destroy()
		button_track_trace_temp.destroy()
		button_3D.destroy()
		button_eng_spec.destroy()
		button_RvE.destroy()
		button_track_angle.destroy()
		button_cnn.destroy()
		button_plot_proj.destroy()
		entry_bandwidth.destroy()
		button_fit.destroy()
		button_fit_multi_params.destroy()
		entry_multi.destroy()
		button_init_trace.destroy()	


	entry_event_num = Entry(root, borderwidth=5, width=11) 
	entry_event_num.default_text = 'Enter Event #'
	entry_event_num.insert(0, entry_event_num.default_text)
	entry_event_num.bind("<FocusIn>", on_focus_in)
	entry_event_num.bind("<FocusOut>", on_focus_out)
	canvas1.create_window(480, 150, window=entry_event_num)

	button_plot = Button(root, text="Show Track w/ Trace", command=plot_track)
	canvas1.create_window(480, 220, window=button_plot)

	button_init_trace = Button(root, text="Initial Guess for Trace Fit", command=fit_init_trace)
	canvas1.create_window(480, 280, window=button_init_trace)

	entry_bandwidth = Entry(root, width=14) 
	entry_bandwidth.default_text = 'Bandwidth Factor'
	entry_bandwidth.insert(0, entry_bandwidth.default_text)
	entry_bandwidth.bind("<FocusIn>", on_focus_in)
	entry_bandwidth.bind("<FocusOut>", on_focus_out)
	canvas1.create_window(480, 340, window=entry_bandwidth)

	button_plot_proj = Button(root, text="Trace from Projection to Fit Line", command=plot_track_proj)
	canvas1.create_window(480, 370, window=button_plot_proj)

	button_fit = Button(root, text="Initial Guess for Projected Trace Fit", command=fit_trace)
	canvas1.create_window(480, 400, window=button_fit)

	entry_multi = Entry(root, width=42) 
	entry_multi.insert(0,'Paste Fit Parameters | Use * in Front of Param to Fix Value')
	entry_multi.bind('<FocusIn>', temp_text61)
	canvas1.create_window(480, 460, window=entry_multi)

	button_fit_multi_params = Button(root, text="Multi-peak Fit from Params", command=plot_spectrum_multi_params)
	canvas1.create_window(480, 490, window=button_fit_multi_params)

	# Track with Trace Button
	button_track_trace_temp = Button(text='Track with Trace',fg='red', command=close_out)
	canvas1.create_window(163, 385, window=button_track_trace_temp)

	# Energy Spectrum Button    
	button_eng_spec = Button(text='Energy Spectrum',fg='green', command=open_eng_spec)
	canvas1.create_window(163, 235, window=button_eng_spec)
	button_eng_spec["state"] = "disabled"

	# Range vs Energy Button
	button_RvE = Button(text='Range vs Energy',fg='green',command=open_RvE)
	canvas1.create_window(163, 285, window=button_RvE)
	button_RvE["state"] = "disabled"

	# 3D Plot Button
	button_3D = Button(text='3D Event Plot',fg='green', command=open_3d_plot)
	canvas1.create_window(163, 335, window=button_3D)
	button_3D["state"] = "disabled"

	# Track Angle Button
	button_track_angle = Button(text='Track Angles',fg='green', command=open_track_angles)
	canvas1.create_window(163, 435, window=button_track_angle)
	button_track_angle["state"] = "disabled"

	# ConvNet Button
	button_cnn = Button(text='ConvNet Track ID',fg='green', command=cnn)
	canvas1.create_window(163, 485, window=button_cnn)
	button_cnn["state"] = "disabled"


################################################################ Track Angles
################################################################

def track_angles():
	from tqdm import tqdm
	from collections import defaultdict
	import matplotlib.colors as colors

	def create_dictionaries(all_angle_list, all_len_list, all_energy_list):
		len_dict = defaultdict(list)
		energy_dict = defaultdict(list)

		print('Creating Length and Energy Dictionaries')
		pbar = tqdm(total=len(all_angle_list))

		for evnt_n, angle in enumerate(all_angle_list):
			index = int(angle)
			len_dict[index].append(all_len_list[evnt_n])
			energy_dict[index].append(all_energy_list[evnt_n])
			pbar.update(n=1)

		return len_dict, energy_dict


	len_dict, energy_dict = create_dictionaries(angle_list, len_list, tot_energy)


	def close_out():
		button_track_angles_temp.destroy()
		button_plot_RvE.destroy()
		entry_bins.destroy()
		button_plot_engSpec.destroy()
		button_plot_RvE_heatmap.destroy()
		button_3D.destroy()
		button_eng_spec.destroy()
		button_RvE.destroy()
		button_cnn.destroy()
		button_track_trace.destroy()
		button_plot_RvE_heatmap_sum.destroy()
		entry_start_deg.destroy()
		entry_end_deg.destroy()

	def plot_RvE(len_dict, energy_dict):
		# Plot range vs integrated charge for 10 degree angle range 
		# Separate into angle ranges by 10 degree increments
		from itertools import chain
		plt.rcParams['figure.figsize'] = [10, 10]
		degree_range = 10

		# Create Plot for 0-10 Degrees
		starting_angle=0
		range_0to10 = []
		int_charge_0to10 = []
		print('Plotting 0-10 Degrees')
		pbar = tqdm(total=10)
		for i in range(degree_range):
			range_0to10.extend(len_dict[starting_angle + i])
			int_charge_0to10.extend(energy_dict[starting_angle + i])
			pbar.update(n=1)
		range_0to10 = [item for sublist in range_0to10 for item in (sublist if hasattr(sublist, '__iter__') else [sublist])]

		int_charge_0to10 = [item for sublist in int_charge_0to10 for item in (sublist if hasattr(sublist, '__iter__') else [sublist])]
		plt.subplot(3, 3, 1)
		plt.scatter(int_charge_0to10, range_0to10)
		plt.title('0 - 10 Degrees', fontdict = {'fontsize' : 10})
		plt.ylabel('Range (mm)', fontdict = {'fontsize' : 10})
		plt.xlabel('Integrated Charge', fontdict = {'fontsize' : 10})
		plt.xticks(fontsize = 10)
		plt.yticks(fontsize = 10)
		del range_0to10
		del int_charge_0to10

		# Create Plot for 10-20 Degrees
		starting_angle=10
		range_10to20 = []
		int_charge_10to20 = []
		print('Plotting 10-20 Degrees')
		pbar = tqdm(total=10)
		for i in range(degree_range):
			range_10to20.extend(len_dict[starting_angle + i])
			int_charge_10to20.extend(energy_dict[starting_angle + i])
			pbar.update(n=1)
		#flat_ls = [item for sublist in int_charge_10to20 for item in sublist]
		range_10to20 = [item for sublist in range_10to20 for item in (sublist if hasattr(sublist, '__iter__') else [sublist])]

		int_charge_10to20 = [item for sublist in int_charge_10to20 for item in (sublist if hasattr(sublist, '__iter__') else [sublist])]
		plt.subplot(3, 3, 2)
		plt.scatter(int_charge_10to20, range_10to20)
		plt.title('10 - 20 Degrees', fontdict = {'fontsize' : 10})
		plt.ylabel('Range (mm)', fontdict = {'fontsize' : 10})
		plt.xlabel('Integrated Charge', fontdict = {'fontsize' : 10})
		plt.xticks(fontsize = 10)
		plt.yticks(fontsize = 10)
		del range_10to20
		del int_charge_10to20

		# Create Plot for 20-30 Degrees
		starting_angle=20
		range_20to30 = []
		int_charge_20to30 = []
		print('Plotting 20-30 Degrees')
		pbar = tqdm(total=10)
		for i in range(degree_range):
			range_20to30.extend(len_dict[starting_angle + i])
			int_charge_20to30.extend(energy_dict[starting_angle + i])
			pbar.update(n=1)
		range_20to30 = [item for sublist in range_20to30 for item in (sublist if hasattr(sublist, '__iter__') else [sublist])]

		int_charge_20to30 = [item for sublist in int_charge_20to30 for item in (sublist if hasattr(sublist, '__iter__') else [sublist])]
		plt.subplot(3, 3, 3)
		plt.scatter(int_charge_20to30, range_20to30)
		plt.title('20 - 30 Degrees', fontdict = {'fontsize' : 10})
		plt.ylabel('Range (mm)', fontdict = {'fontsize' : 10})
		plt.xlabel('Integrated Charge', fontdict = {'fontsize' : 10})
		plt.xticks(fontsize = 10)
		plt.yticks(fontsize = 10)
		del range_20to30
		del int_charge_20to30

		# Create Plot for 30-40 Degrees
		starting_angle=30
		range_30to40 = []
		int_charge_30to40 = []
		print('Plotting 30-40 Degrees')
		pbar = tqdm(total=10)
		for i in range(degree_range):
			range_30to40.extend(len_dict[starting_angle + i])
			int_charge_30to40.extend(energy_dict[starting_angle + i])
			pbar.update(n=1)
		range_30to40 = [item for sublist in range_30to40 for item in (sublist if hasattr(sublist, '__iter__') else [sublist])]

		int_charge_30to40 = [item for sublist in int_charge_30to40 for item in (sublist if hasattr(sublist, '__iter__') else [sublist])]
		plt.subplot(3, 3, 4)
		plt.scatter(int_charge_30to40, range_30to40)
		plt.title('30 - 40 Degrees', fontdict = {'fontsize' :  10})
		plt.ylabel('Range (mm)', fontdict = {'fontsize' : 10})
		plt.xlabel('Integrated Charge', fontdict = {'fontsize' : 10})
		plt.xticks(fontsize = 10)
		plt.yticks(fontsize = 10)
		del range_30to40
		del int_charge_30to40

		# Create Plot for 40-50 Degrees
		starting_angle=40
		range_40to50 = []
		int_charge_40to50 = []
		print('Plotting 40-50 Degrees')
		pbar = tqdm(total=10)
		for i in range(degree_range):
			range_40to50.extend(len_dict[starting_angle + i])
			int_charge_40to50.extend(energy_dict[starting_angle + i])
			pbar.update(n=1)
		range_40to50 = [item for sublist in range_40to50 for item in (sublist if hasattr(sublist, '__iter__') else [sublist])]

		int_charge_40to50 = [item for sublist in int_charge_40to50 for item in (sublist if hasattr(sublist, '__iter__') else [sublist])]
		plt.subplot(3, 3, 5)
		plt.scatter(int_charge_40to50, range_40to50)
		plt.title('40 - 50 Degrees', fontdict = {'fontsize' : 10})
		plt.ylabel('Range (mm)', fontdict = {'fontsize' : 10})
		plt.xlabel('Integrated Charge', fontdict = {'fontsize' : 10})
		plt.xticks(fontsize = 10)
		plt.yticks(fontsize = 10)
		del range_40to50
		del int_charge_40to50

		# Create Plot for 50-60 Degrees
		starting_angle=50
		range_50to60 = []
		int_charge_50to60 = []
		print('Plotting 50-60 Degrees')
		pbar = tqdm(total=10)
		for i in range(degree_range):
			range_50to60.extend(len_dict[starting_angle + i])
			int_charge_50to60.extend(energy_dict[starting_angle + i])
			pbar.update(n=1)
		range_50to60 = [item for sublist in range_50to60 for item in (sublist if hasattr(sublist, '__iter__') else [sublist])]

		int_charge_50to60 = [item for sublist in int_charge_50to60 for item in (sublist if hasattr(sublist, '__iter__') else [sublist])]
		plt.subplot(3, 3, 6)
		plt.scatter(int_charge_50to60, range_50to60)
		plt.title('50 - 60 Degrees', fontdict = {'fontsize' : 10})
		plt.ylabel('Range (mm)', fontdict = {'fontsize' : 10})
		plt.xlabel('Integrated Charge', fontdict = {'fontsize' : 10})
		plt.xticks(fontsize = 10)
		plt.yticks(fontsize = 10)
		del range_50to60
		del int_charge_50to60

		# Create Plot for 60-70 Degrees
		starting_angle=60
		range_60to70 = []
		int_charge_60to70 = []
		print('Plotting 60-70 Degrees')
		pbar = tqdm(total=10)
		for i in range(degree_range):
			range_60to70.extend(len_dict[starting_angle + i])
			int_charge_60to70.extend(energy_dict[starting_angle + i])
			pbar.update(n=1)
		range_60to70 = [item for sublist in range_60to70 for item in (sublist if hasattr(sublist, '__iter__') else [sublist])]

		int_charge_60to70 = [item for sublist in int_charge_60to70 for item in (sublist if hasattr(sublist, '__iter__') else [sublist])]
		plt.subplot(3, 3, 7)
		plt.scatter(int_charge_60to70, range_60to70)
		plt.title('60 - 70 Degrees', fontdict = {'fontsize' : 10})
		plt.ylabel('Range (mm)', fontdict = {'fontsize' : 10})
		plt.xlabel('Integrated Charge', fontdict = {'fontsize' : 10})
		plt.xticks(fontsize = 10)
		plt.yticks(fontsize = 10)
		del range_60to70
		del int_charge_60to70

		# Create Plot for 70-80 Degrees
		starting_angle=70
		range_70to80 = []
		int_charge_70to80 = []
		print('Plotting 70-80 Degrees')
		pbar = tqdm(total=10)
		for i in range(degree_range):
			range_70to80.extend(len_dict[starting_angle + i])
			int_charge_70to80.extend(energy_dict[starting_angle + i])
			pbar.update(n=1)
		range_70to80 = [item for sublist in range_70to80 for item in (sublist if hasattr(sublist, '__iter__') else [sublist])]

		int_charge_70to80 = [item for sublist in int_charge_70to80 for item in (sublist if hasattr(sublist, '__iter__') else [sublist])]
		plt.subplot(3, 3, 8)
		plt.scatter(int_charge_70to80, range_70to80)
		plt.title('70 - 80 Degrees', fontdict = {'fontsize' : 10})
		plt.ylabel('Range (mm)', fontdict = {'fontsize' : 10})
		plt.xlabel('Integrated Charge', fontdict = {'fontsize' : 10})
		plt.xticks(fontsize = 10)
		plt.yticks(fontsize = 10)
		del range_70to80
		del int_charge_70to80

		# Create Plot for 80-90 Degrees
		starting_angle=80
		range_80to90 = []
		int_charge_80to90 = []
		print('Plotting 80-90 Degrees')
		pbar = tqdm(total=10)
		for i in range(degree_range):
			range_80to90.extend(len_dict[starting_angle + i])
			int_charge_80to90.extend(energy_dict[starting_angle + i])
			pbar.update(n=1)
		range_80to90 = [item for sublist in range_80to90 for item in (sublist if hasattr(sublist, '__iter__') else [sublist])]

		int_charge_80to90 = [item for sublist in int_charge_80to90 for item in (sublist if hasattr(sublist, '__iter__') else [sublist])]
		plt.subplot(3, 3, 9)
		plt.scatter(int_charge_80to90, range_80to90)
		plt.title('80 - 90 Degrees', fontdict = {'fontsize' : 10})
		plt.ylabel('Range (mm)', fontdict = {'fontsize' : 10})
		plt.xlabel('Integrated Charge', fontdict = {'fontsize' : 10})
		plt.xticks(fontsize = 10)
		plt.yticks(fontsize = 10)
		del range_80to90
		del int_charge_80to90

		plt.tight_layout()
		plt.show()


	def plot_RvE_heatmap(len_dict, energy_dict):
		# Plot range vs integrated charge for 10 degree angle range 
		from itertools import chain

		num_bins = int(entry_bins.get())
		plt.rcParams['figure.figsize'] = [10, 10]
		degree_range = 10

		# Create Plot for 0-10 Degrees
		starting_angle=0
		range_0to10 = []
		int_charge_0to10 = []
		pbar = tqdm(total=10)
		for i in range(degree_range):
			range_0to10.extend(len_dict[starting_angle + i])
			int_charge_0to10.extend(energy_dict[starting_angle + i])
			pbar.update(n=1)
		range_0to10 = [item for sublist in range_0to10 for item in (sublist if hasattr(sublist, '__iter__') else [sublist])]

		int_charge_0to10 = [item for sublist in int_charge_0to10 for item in (sublist if hasattr(sublist, '__iter__') else [sublist])]
		if len(int_charge_0to10) > 0:
			plt.rcParams['figure.figsize'] = [10, 10]
			plt.subplot(3, 3, 1)
			plt.hist2d(np.asarray(int_charge_0to10), np.asarray(range_0to10), (num_bins, num_bins), cmap=plt.cm.jet, norm=colors.LogNorm())
			plt.colorbar()
			plt.title('0 - 10 Degrees', fontdict = {'fontsize' : 10})
			plt.ylabel('Range (mm)', fontdict = {'fontsize' : 10})
			plt.xlabel('Integrated Charge', fontdict = {'fontsize' : 10})
			plt.xticks(fontsize = 10)
			plt.yticks(fontsize = 10)
		del range_0to10
		del int_charge_0to10

		# Create Plot for 10-20 Degrees
		starting_angle=10
		range_10to20 = []
		int_charge_10to20 = []
		pbar = tqdm(total=10)
		for i in range(degree_range):
			range_10to20.extend(len_dict[starting_angle + i])
			int_charge_10to20.extend(energy_dict[starting_angle + i])
			pbar.update(n=1)
		range_10to20 = [item for sublist in range_10to20 for item in (sublist if hasattr(sublist, '__iter__') else [sublist])]

		int_charge_10to20 = [item for sublist in int_charge_10to20 for item in (sublist if hasattr(sublist, '__iter__') else [sublist])]
		if len(int_charge_10to20) > 0:
			plt.subplot(3, 3, 2)
			plt.hist2d(np.asarray(int_charge_10to20), np.asarray(range_10to20), (num_bins, num_bins), cmap=plt.cm.jet, norm=colors.LogNorm())
			plt.colorbar()
			plt.title('10 - 20 Degrees', fontdict = {'fontsize' : 10})
			plt.ylabel('Range (mm)', fontdict = {'fontsize' : 10})
			plt.xlabel('Integrated Charge', fontdict = {'fontsize' : 10})
			plt.xticks(fontsize = 10)
			plt.yticks(fontsize = 10)
		del range_10to20
		del int_charge_10to20

		# Create Plot for 20-30 Degrees
		starting_angle=20
		range_20to30 = []
		int_charge_20to30 = []
		pbar = tqdm(total=10)
		for i in range(degree_range):
			range_20to30.extend(len_dict[starting_angle + i])
			int_charge_20to30.extend(energy_dict[starting_angle + i])
			pbar.update(n=1)
		range_20to30 = [item for sublist in range_20to30 for item in (sublist if hasattr(sublist, '__iter__') else [sublist])]

		int_charge_20to30 = [item for sublist in int_charge_20to30 for item in (sublist if hasattr(sublist, '__iter__') else [sublist])]
		if len(int_charge_20to30) > 0:
			plt.subplot(3, 3, 3)
			plt.hist2d(np.asarray(int_charge_20to30), np.asarray(range_20to30), (num_bins, num_bins), cmap=plt.cm.jet, norm=colors.LogNorm())
			plt.colorbar()
			plt.title('20 - 30 Degrees', fontdict = {'fontsize' : 10})
			plt.ylabel('Range (mm)', fontdict = {'fontsize' : 10})
			plt.xlabel('Integrated Charge', fontdict = {'fontsize' : 10})
			plt.xticks(fontsize = 10)
			plt.yticks(fontsize = 10)
		del range_20to30
		del int_charge_20to30

		# Create Plot for 30-40 Degrees
		starting_angle=30
		range_30to40 = []
		int_charge_30to40 = []
		pbar = tqdm(total=10)
		for i in range(degree_range):
			range_30to40.extend(len_dict[starting_angle + i])
			int_charge_30to40.extend(energy_dict[starting_angle + i])
			pbar.update(n=1)
		range_30to40 = [item for sublist in range_30to40 for item in (sublist if hasattr(sublist, '__iter__') else [sublist])]

		int_charge_30to40 = [item for sublist in int_charge_30to40 for item in (sublist if hasattr(sublist, '__iter__') else [sublist])]
		if len(int_charge_30to40) > 0:
			plt.subplot(3, 3, 4)
			plt.hist2d(np.asarray(int_charge_30to40), np.asarray(range_30to40), (num_bins, num_bins), cmap=plt.cm.jet, norm=colors.LogNorm())
			plt.colorbar()
			plt.title('30 - 40 Degrees', fontdict = {'fontsize' :  10})
			plt.ylabel('Range (mm)', fontdict = {'fontsize' : 10})
			plt.xlabel('Integrated Charge', fontdict = {'fontsize' : 10})
			plt.xticks(fontsize = 10)
			plt.yticks(fontsize = 10)
		del range_30to40
		del int_charge_30to40

		# Create Plot for 40-50 Degrees
		starting_angle=40
		range_40to50 = []
		int_charge_40to50 = []
		pbar = tqdm(total=10)
		for i in range(degree_range):
			range_40to50.extend(len_dict[starting_angle + i])
			int_charge_40to50.extend(energy_dict[starting_angle + i])
			pbar.update(n=1)
		range_40to50 = [item for sublist in range_40to50 for item in (sublist if hasattr(sublist, '__iter__') else [sublist])]

		int_charge_40to50 = [item for sublist in int_charge_40to50 for item in (sublist if hasattr(sublist, '__iter__') else [sublist])]
		if len(int_charge_40to50) > 0:
			plt.subplot(3, 3, 5)
			plt.hist2d(np.asarray(int_charge_40to50), np.asarray(range_40to50), (num_bins, num_bins), cmap=plt.cm.jet, norm=colors.LogNorm())
			plt.colorbar()
			plt.title('40 - 50 Degrees', fontdict = {'fontsize' : 10})
			plt.ylabel('Range (mm)', fontdict = {'fontsize' : 10})
			plt.xlabel('Integrated Charge', fontdict = {'fontsize' : 10})
			plt.xticks(fontsize = 10)
			plt.yticks(fontsize = 10)
		del range_40to50
		del int_charge_40to50

		# Create Plot for 50-60 Degrees
		starting_angle=50
		range_50to60 = []
		int_charge_50to60 = []
		pbar = tqdm(total=10)
		for i in range(degree_range):
			range_50to60.extend(len_dict[starting_angle + i])
			int_charge_50to60.extend(energy_dict[starting_angle + i])
			pbar.update(n=1)
		range_50to60 = [item for sublist in range_50to60 for item in (sublist if hasattr(sublist, '__iter__') else [sublist])]

		int_charge_50to60 = [item for sublist in int_charge_50to60 for item in (sublist if hasattr(sublist, '__iter__') else [sublist])]
		if len(int_charge_50to60) > 0:
			plt.subplot(3, 3, 6)
			plt.hist2d(np.asarray(int_charge_50to60), np.asarray(range_50to60), (num_bins, num_bins), cmap=plt.cm.jet, norm=colors.LogNorm())
			plt.colorbar()
			plt.title('50 - 60 Degrees', fontdict = {'fontsize' : 10})
			plt.ylabel('Range (mm)', fontdict = {'fontsize' : 10})
			plt.xlabel('Integrated Charge', fontdict = {'fontsize' : 10})
			plt.xticks(fontsize = 10)
			plt.yticks(fontsize = 10)
		del range_50to60
		del int_charge_50to60

		# Create Plot for 60-70 Degrees
		starting_angle=60
		range_60to70 = []
		int_charge_60to70 = []
		pbar = tqdm(total=10)
		for i in range(degree_range):
			range_60to70.extend(len_dict[starting_angle + i])
			int_charge_60to70.extend(energy_dict[starting_angle + i])
			pbar.update(n=1)
		range_60to70 = [item for sublist in range_60to70 for item in (sublist if hasattr(sublist, '__iter__') else [sublist])]

		int_charge_60to70 = [item for sublist in int_charge_60to70 for item in (sublist if hasattr(sublist, '__iter__') else [sublist])]
		if len(int_charge_60to70) > 0:
			plt.subplot(3, 3, 7)
			plt.hist2d(np.asarray(int_charge_60to70), np.asarray(range_60to70), (num_bins, num_bins), cmap=plt.cm.jet, norm=colors.LogNorm())
			plt.colorbar()
			plt.title('60 - 70 Degrees', fontdict = {'fontsize' : 10})
			plt.ylabel('Range (mm)', fontdict = {'fontsize' : 10})
			plt.xlabel('Integrated Charge', fontdict = {'fontsize' : 10})
			plt.xticks(fontsize = 10)
			plt.yticks(fontsize = 10)
		del range_60to70
		del int_charge_60to70

		# Create Plot for 70-80 Degrees
		starting_angle=70
		range_70to80 = []
		int_charge_70to80 = []
		pbar = tqdm(total=10)
		for i in range(degree_range):
			range_70to80.extend(len_dict[starting_angle + i])
			int_charge_70to80.extend(energy_dict[starting_angle + i])
			pbar.update(n=1)
		range_70to80 = [item for sublist in range_70to80 for item in (sublist if hasattr(sublist, '__iter__') else [sublist])]

		int_charge_70to80 = [item for sublist in int_charge_70to80 for item in (sublist if hasattr(sublist, '__iter__') else [sublist])]
		if len(int_charge_70to80) > 0:
			plt.subplot(3, 3, 8)
			plt.hist2d(np.asarray(int_charge_70to80), np.asarray(range_70to80), (num_bins, num_bins), cmap=plt.cm.jet, norm=colors.LogNorm())
			plt.colorbar()
			plt.title('70 - 80 Degrees', fontdict = {'fontsize' : 10})
			plt.ylabel('Range (mm)', fontdict = {'fontsize' : 10})
			plt.xlabel('Integrated Charge', fontdict = {'fontsize' : 10})
			plt.xticks(fontsize = 10)
			plt.yticks(fontsize = 10)
		del range_70to80
		del int_charge_70to80

		# Create Plot for 80-90 Degrees
		starting_angle=80
		range_80to90 = []
		int_charge_80to90 = []
		pbar = tqdm(total=10)
		for i in range(degree_range):
			range_80to90.extend(len_dict[starting_angle + i])
			int_charge_80to90.extend(energy_dict[starting_angle + i])
			pbar.update(n=1)
		range_80to90 = [item for sublist in range_80to90 for item in (sublist if hasattr(sublist, '__iter__') else [sublist])]

		int_charge_80to90 = [item for sublist in int_charge_80to90 for item in (sublist if hasattr(sublist, '__iter__') else [sublist])]
		if len(int_charge_80to90) > 0:
			plt.subplot(3, 3, 9)
			plt.hist2d(np.asarray(int_charge_80to90), np.asarray(range_80to90), (num_bins, num_bins), cmap=plt.cm.jet, norm=colors.LogNorm())
			plt.colorbar()
			plt.title('80 - 90 Degrees', fontdict = {'fontsize' : 10})
			plt.ylabel('Range (mm)', fontdict = {'fontsize' : 10})
			plt.xlabel('Integrated Charge', fontdict = {'fontsize' : 10})
			plt.xticks(fontsize = 10)
			plt.yticks(fontsize = 10)
		del range_80to90
		del int_charge_80to90
	
		plt.tight_layout()
		plt.show()


	def plot_engSpec(energy_dict):
		from itertools import chain
		plt.rcParams['figure.figsize'] = [10, 10]
		bins = int(entry_bins.get())
		degree_range = 10

		# Create Plot for 0-10 Degrees
		starting_angle=0
		int_charge_0to10 = []
		pbar = tqdm(total=10)
		for i in range(degree_range):
			int_charge_0to10.extend(energy_dict[starting_angle + i])
			pbar.update(n=1)
		int_charge_0to10 = [item for sublist in int_charge_0to10 for item in (sublist if hasattr(sublist, '__iter__') else [sublist])]
		plt.subplot(3, 3, 1)
		plt.hist(int_charge_0to10, bins)
		plt.title('0 - 10 Degrees')
		plt.ylabel('counts')
		plt.xlabel('Integrated Charge')
		del int_charge_0to10

		# Create Plot for 10-20 Degrees
		starting_angle=10
		int_charge_10to20 = []
		pbar = tqdm(total=10)
		for i in range(degree_range):
			int_charge_10to20.extend(energy_dict[starting_angle + i])
			pbar.update(n=1)
		int_charge_10to20 = [item for sublist in int_charge_10to20 for item in (sublist if hasattr(sublist, '__iter__') else [sublist])]
		plt.subplot(3, 3, 2)
		plt.hist(int_charge_10to20, bins)
		plt.title('10 - 20 Degrees')
		plt.ylabel('counts')
		plt.xlabel('Integrated Charge')
		del int_charge_10to20

		# Create Plot for 20-30 Degrees
		starting_angle=20
		int_charge_20to30 = []
		pbar = tqdm(total=10)
		for i in range(degree_range):
			int_charge_20to30.extend(energy_dict[starting_angle + i])
			pbar.update(n=1)
		int_charge_20to30 = [item for sublist in int_charge_20to30 for item in (sublist if hasattr(sublist, '__iter__') else [sublist])]
		plt.subplot(3, 3, 3)
		plt.hist(int_charge_20to30, bins)
		plt.title('20 - 30 Degrees')
		plt.ylabel('counts')
		plt.xlabel('Integrated Charge')
		del int_charge_20to30

		# Create Plot for 30-40 Degrees
		starting_angle=30
		int_charge_30to40 = []
		pbar = tqdm(total=10)
		for i in range(degree_range):
			int_charge_30to40.extend(energy_dict[starting_angle + i])
			pbar.update(n=1)
		int_charge_30to40 = [item for sublist in int_charge_30to40 for item in (sublist if hasattr(sublist, '__iter__') else [sublist])]
		plt.subplot(3, 3, 4)
		plt.hist(int_charge_30to40, bins)
		plt.title('30 - 40 Degrees')
		plt.ylabel('counts')
		plt.xlabel('Integrated Charge')
		del int_charge_30to40

		# Create Plot for 40-50 Degrees
		starting_angle=40
		int_charge_40to50 = []
		pbar = tqdm(total=10)
		for i in range(degree_range):
			int_charge_40to50.extend(energy_dict[starting_angle + i])
			pbar.update(n=1)
		int_charge_40to50 = [item for sublist in int_charge_40to50 for item in (sublist if hasattr(sublist, '__iter__') else [sublist])]
		plt.subplot(3, 3, 5)
		plt.hist(int_charge_40to50, bins)
		plt.title('40 - 50 Degrees')
		plt.ylabel('Range (mm)')
		plt.xlabel('Integrated Charge')
		del int_charge_40to50

		# Create Plot for 50-60 Degrees
		starting_angle=50
		int_charge_50to60 = []
		pbar = tqdm(total=10)
		for i in range(degree_range):
			int_charge_50to60.extend(energy_dict[starting_angle + i])
			pbar.update(n=1)
		int_charge_50to60 = [item for sublist in int_charge_50to60 for item in (sublist if hasattr(sublist, '__iter__') else [sublist])]
		plt.subplot(3, 3, 6)
		plt.hist(int_charge_50to60, bins)
		plt.title('50 - 60 Degrees')
		plt.ylabel('counts')
		plt.xlabel('Integrated Charge')
		del int_charge_50to60

		# Create Plot for 60-70 Degrees
		starting_angle=60
		int_charge_60to70 = []
		pbar = tqdm(total=10)
		for i in range(degree_range):
			int_charge_60to70.extend(energy_dict[starting_angle + i])
			pbar.update(n=1)
		int_charge_60to70 = [item for sublist in int_charge_60to70 for item in (sublist if hasattr(sublist, '__iter__') else [sublist])]
		plt.subplot(3, 3, 7)
		plt.hist(int_charge_60to70, bins)
		plt.title('60 - 70 Degrees')
		plt.ylabel('counts')
		plt.xlabel('Integrated Charge')
		del int_charge_60to70

		# Create Plot for 70-80 Degrees
		starting_angle=70
		int_charge_70to80 = []
		pbar = tqdm(total=10)
		for i in range(degree_range):
			int_charge_70to80.extend(energy_dict[starting_angle + i])
			pbar.update(n=1)
		int_charge_70to80 = [item for sublist in int_charge_70to80 for item in (sublist if hasattr(sublist, '__iter__') else [sublist])]
		plt.subplot(3, 3, 8)
		plt.hist(int_charge_70to80, bins)
		plt.title('70 - 80 Degrees')
		plt.ylabel('counts')
		plt.xlabel('Integrated Charge')
		del int_charge_70to80

		# Create Plot for 80-90 Degrees
		starting_angle=80
		int_charge_80to90 = []
		pbar = tqdm(total=10)
		for i in range(degree_range):
			int_charge_80to90.extend(energy_dict[starting_angle + i])
			pbar.update(n=1)
		int_charge_80to90 = [item for sublist in int_charge_80to90 for item in (sublist if hasattr(sublist, '__iter__') else [sublist])]
		plt.subplot(3, 3, 9)
		plt.hist(int_charge_80to90,bins)
		plt.title('80 - 90 Degrees')
		plt.ylabel('counts')
		plt.xlabel('Integrated Charge')
		del int_charge_80to90

		plt.tight_layout()
		plt.show()


	def plot_RvE_heatmap_sum(len_dict, energy_dict):
		# Plot range vs integrated charge for degree angle range from start_deg to end_deg
		from itertools import chain
		num_bins = int(entry_bins.get())
		start_deg = int(entry_start_deg.get())
		end_deg = int(entry_end_deg.get())
		if end_deg == 90:
        		end_deg = max(len_dict.keys())
		plt.rcParams['figure.figsize'] = [10, 10]
		degree_range = end_deg - start_deg + 1

		# Create Plot for start_deg-end_deg Degrees
		range_deg = []
		int_charge_deg = []
		pbar = tqdm(total=degree_range)
		for i in range(start_deg, end_deg+1):
			range_deg.extend(len_dict[i])
			int_charge_deg.extend(energy_dict[i])
			pbar.update(n=1)
		range_deg = [item for sublist in range_deg for item in (sublist if hasattr(sublist, '__iter__') else [sublist])]
		int_charge_deg = [item for sublist in int_charge_deg for item in (sublist if hasattr(sublist, '__iter__') else [sublist])]
		if len(int_charge_deg) > 0:
			plt.hist2d(np.asarray(int_charge_deg), np.asarray(range_deg), (num_bins, num_bins), cmap=plt.cm.jet, norm=colors.LogNorm())
			plt.colorbar()
			plt.title('{} - {} Degrees'.format(start_deg, end_deg), fontdict = {'fontsize' : 10})
			plt.ylabel('Range (mm)', fontdict = {'fontsize' : 10})
			plt.xlabel('Integrated Charge', fontdict = {'fontsize' : 10})
			plt.xticks(fontsize = 10)
			plt.yticks(fontsize = 10)

		plt.tight_layout()
		plt.show()


	def temp_text7(e):
		entry_bins.delete(0,"end")

	def temp_text8(e):
		entry_bins_heat.delete(0,"end")

	def temp_text81(e):
		entry_bins_heat_sum.delete(0,"end")

	def temp_text82(e):
		entry_start_deg.delete(0,"end")

	def temp_text83(e):
		entry_end_deg.delete(0,"end")


	def on_focus_in(event):
		if event.widget.get() == event.widget.default_text:
			event.widget.delete(0, END)

	def on_focus_out(event):
		if event.widget.get() == '':
			event.widget.insert(0, event.widget.default_text)


	entry_bins = Entry(root, borderwidth=5, width=12)
	entry_bins.default_text = 'Enter # of Bins'
	entry_bins.insert(0, entry_bins.default_text)
	entry_bins.bind('<FocusIn>', on_focus_in)
	entry_bins.bind('<FocusOut>', on_focus_out)
	canvas1.create_window(480, 155, window=entry_bins)

	button_plot_engSpec = Button(root, text="Energy Spec (10 degree angles)", command=lambda: plot_engSpec(energy_dict))
	canvas1.create_window(480, 225, window=button_plot_engSpec)
	
	button_plot_RvE = Button(root, text="RvE Scatter Plot (10 degree angles)", command=lambda: plot_RvE(len_dict, energy_dict))
	canvas1.create_window(480, 285, window=button_plot_RvE)

	button_plot_RvE_heatmap = Button(root, text="RvE Heatmap (10 degree angles)", command=lambda: plot_RvE_heatmap(len_dict, energy_dict))
	canvas1.create_window(480, 345, window=button_plot_RvE_heatmap)


	button_track_angles_temp = Button(text='Track Angles',fg='red', command=close_out)
	canvas1.create_window(163, 435, window=button_track_angles_temp)

	entry_start_deg = Entry(root, width=7) 
	entry_start_deg.insert(0, 'Start Deg')
	entry_start_deg.bind("<FocusIn>", temp_text82)
	canvas1.create_window(450, 405, height=21, window=entry_start_deg)

	entry_end_deg = Entry(root, width=7) 
	entry_end_deg.insert(0, 'End Deg')
	entry_end_deg.bind("<FocusIn>", temp_text83)
	canvas1.create_window(510, 405, height=21, window=entry_end_deg)

	button_plot_RvE_heatmap_sum = Button(root, text="RvE Heatmap for Specific Angle Range", command=lambda: plot_RvE_heatmap_sum(len_dict, energy_dict))
	canvas1.create_window(480, 435, window=button_plot_RvE_heatmap_sum)

	# Energy Spectrum Button    
	button_eng_spec = Button(text='Energy Spectrum',fg='green', command=open_eng_spec)
	canvas1.create_window(163, 235, window=button_eng_spec)
	button_eng_spec["state"] = "disabled"

	# Range vs Energy Button
	button_RvE = Button(text='Range vs Energy',fg='green',command=open_RvE)
	canvas1.create_window(163, 285, window=button_RvE)
	button_RvE["state"] = "disabled"

	# 3D Plot Button
	button_3D = Button(text='3D Event Plot',fg='green', command=open_3d_plot)
	canvas1.create_window(163, 335, window=button_3D)
	button_3D["state"] = "disabled"

	# Track with Trace Button
	button_track_trace = Button(text='Track with Trace',fg='green', command=open_track_trace)
	canvas1.create_window(163, 385, window=button_track_trace)
	button_track_trace["state"] = "disabled"

	# ConvNet Button
	button_cnn = Button(text='ConvNet Track ID',fg='green', command=cnn)
	canvas1.create_window(163, 485, window=button_cnn)
	button_cnn["state"] = "disabled"
	


################################################################ CNN
################################################################

def cnn():

	def prev_cut():
		import glob
		# All files and directories ending with .txt and that don't begin with a dot:
		prev_cut_path = os.path.join(sub_mymainpath, "*.jpg")
		global init_image_list
		init_image_list = glob.glob(prev_cut_path)
		image_list = []
		for i in range(len(init_image_list)):
			image_list.append(ImageTk.PhotoImage(Image.open(init_image_list[i])))


		newWindow = Toplevel(root)
		newWindow.title('Image Viewer')
		newWindow.geometry("1000x1000")

		global my_label
		my_label = Label(newWindow, image=image_list[0])
		my_label.place(x=0, y=0)
		
		global image_index 
		image_index = 0

		def forward(image_number):
			global my_label
			global button_forward
			global button_back
			image_index = image_number - 1

			my_label.place_forget()
			my_label = Label(newWindow, image=image_list[image_number-1])
			button_forward = Button(newWindow, text=">>", command=lambda: forward(image_number+1))
			button_exit = Button(newWindow, text="Exit Program", command=newWindow.destroy)
			button_back = Button(newWindow, text="<<", command=lambda: back(image_number-1))
			button_select = Button(newWindow, text="Select Cut and Deploy Model", command=lambda: select_cut(image_index, init_image_list))
			button_select_only = Button(newWindow, text="Select Cut", command=lambda: select_cut_only(image_index, init_image_list))
			
			if image_number == len(init_image_list):
				button_forward = Button(newWindow, text=">>", state=DISABLED)

			my_label.place(x=0, y=0)
			button_back.place(x=350, y=40)
			button_exit.place(x=470, y=80)
			button_select.place(x=430, y=40)
			button_forward.place(x=635, y=40)
			button_select_only.place(x=475, y=0)

		def back(image_number):
			global my_label
			global button_forward
			global button_back
			image_index = image_number - 1

			my_label.place_forget()
			my_label = Label(newWindow, image=image_list[image_number-1])
			button_forward = Button(newWindow, text=">>", command=lambda: forward(image_number+1))
			button_exit = Button(newWindow, text="Exit Program", command=newWindow.destroy)
			button_back = Button(newWindow, text="<<", command=lambda: back(image_number-1))
			button_select = Button(newWindow, text="Select Cut and Deploy Model", command=lambda: select_cut(image_index, init_image_list))
			button_select_only = Button(newWindow, text="Select Cut", command=lambda: select_cut_only(image_index, init_image_list))

			if image_number == 1:
				button_back = Button(newWindow, text="<<", state=DISABLED)

	
			my_label.place(x=0, y=0)
			button_back.place(x=350, y=40)
			button_exit.place(x=470, y=80)
			button_select.place(x=430, y=40)
			button_forward.place(x=635, y=40)
			button_select_only.place(x=475, y=0)

		def select_cut(image_index, init_image_list):
			from tqdm import tqdm
			from PIL import Image, ImageTk, ImageOps
			print('Selected DIRECTORY \n', init_image_list[image_index][:-4])
			global dir_select
			dir_select = init_image_list[image_index][:-4]
			newWindow.destroy()
			
			deploy_model()


		def select_cut_only(image_index, init_image_list):
			from tqdm import tqdm
			from PIL import Image, ImageTk, ImageOps
			print('Selected DIRECTORY \n', init_image_list[image_index][:-4])
			global dir_select
			dir_select = init_image_list[image_index][:-4]
			newWindow.destroy()
			

			
		button_back = Button(newWindow, text="<<", command=back, state=DISABLED)
		button_exit = Button(newWindow, text="Exit Program", command=newWindow.destroy)
		button_forward = Button(newWindow, text=">>", command=lambda: forward(2))
		button_select = Button(newWindow, text="Select Cut and Deploy Model", command=lambda: select_cut(image_index, init_image_list))
		button_select_only = Button(newWindow, text="Select Cut", command=lambda: select_cut_only(image_index, init_image_list))


		button_back.place(x=350, y=40)
		button_exit.place(x=470, y=80)
		button_select.place(x=430, y=40)
		button_forward.place(x=635, y=40)
		button_select_only.place(x=475, y=0)
		
		return


	def select_model():
		mypath = f"/mnt/analysis/e21072/models"
		global model_paths
		model_paths = list(filedialog.askopenfilenames(initialdir=mypath, title="Select a Model"))


	def select_data():
		print('Selecting Data')
		prev_cut()

		button_anom["state"] = "normal"
		button_indv["state"] = "normal"
		button_two["state"] = "normal"
		
	def deploy_model():
		predict(model_paths)


	def close_out():
		button_select.destroy()
		button_cnn_temp.destroy()
		button_data.destroy()
		button_3D.destroy()
		button_eng_spec.destroy()
		button_RvE.destroy()
		button_track_trace.destroy()
		button_track_angle.destroy()
		button_anom.destroy()
		button_indv.destroy()
		button_two.destroy()

	
	def on_focus_in(event):
		if event.widget.get() == event.widget.default_text:
			event.widget.delete(0, END)

	def on_focus_out(event):
		if event.widget.get() == '':
			event.widget.insert(0, event.widget.default_text)

	def view_predictions0():
		pred_text = f'class_0_images.txt'
		print('\nANOMALOUS EVENTS')
		CNN_select_cut(pred_text)

	def view_predictions1():
		pred_text = f'class_1_images.txt'
		print('\nINDIVIDUAL PROTONS/ALPHAS')
		CNN_select_cut(pred_text)

	def view_predictions2():
		pred_text = f'class_2_images.txt'
		print('\nTWO PARTICLE EVENTS')
		CNN_select_cut(pred_text)
	

	button_select = Button(root, text="Select Model", command=select_model)
	canvas1.create_window(435, 305, window=button_select)

	button_data = Button(root, text="Select Data", command=select_data)
	canvas1.create_window(535, 305, window=button_data)

	button_anom = Button(root, text="Anomalous Events", command=view_predictions0)
	canvas1.create_window(350, 365, window=button_anom)
	button_anom["state"] = "disabled"

	button_indv = Button(root, text="Individual Events", command=view_predictions1)
	canvas1.create_window(480, 365, window=button_indv)
	button_indv["state"] = "disabled"

	button_two = Button(root, text="Two Particle Events", command=view_predictions2)
	canvas1.create_window(615, 365, window=button_two)
	button_two["state"] = "disabled"	

	button_cnn_temp = Button(text='ConvNet Track ID',fg='red', command=close_out)
	canvas1.create_window(163, 485, window=button_cnn_temp)

	# Energy Spectrum Button    
	button_eng_spec = Button(text='Energy Spectrum',fg='green', command=open_eng_spec)
	canvas1.create_window(163, 235, window=button_eng_spec)
	button_eng_spec["state"] = "disabled"

	# Range vs Energy Button
	button_RvE = Button(text='Range vs Energy',fg='green',command=open_RvE)
	canvas1.create_window(163, 285, window=button_RvE)
	button_RvE["state"] = "disabled"

	# 3D Plot Button
	button_3D = Button(text='3D Event Plot',fg='green', command=open_3d_plot)
	canvas1.create_window(163, 335, window=button_3D)
	button_3D["state"] = "disabled"

	# Track with Trace Button
	button_track_trace = Button(text='Track with Trace',fg='green', command=open_track_trace)
	canvas1.create_window(163, 385, window=button_track_trace)
	button_track_trace["state"] = "disabled"

	# Track Angle Button
	button_track_angle = Button(text='Track Angles',fg='green', command=open_track_angles)
	canvas1.create_window(163, 435, window=button_track_angle)
	button_track_angle["state"] = "disabled"
		

def predict(model_paths): 
	import torch
	import torch.nn as nn
	from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset, Subset
	from torchvision import transforms, models, datasets
	import torchvision
	import os
	import glob
	from tqdm import tqdm
	import matplotlib.pyplot as plt
	import numpy as np
	import math
	from matplotlib.ticker import MaxNLocator
	from sklearn.metrics import confusion_matrix
	import seaborn as sns
	import pandas as pd
	from torch.optim.lr_scheduler import ReduceLROnPlateau
	from torchvision.transforms import InterpolationMode
	from collections import defaultdict, Counter
	from PIL import Image
	from torch.autograd import Variable
	from scipy import stats

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Simple Identity class that let's input pass without changes
	class Identity(nn.Module):
		def __init__(self):
			super(Identity, self).__init__()

		def forward(self, x):
			return x
	
	class ImageFolderWithPaths(datasets.ImageFolder):
		def __getitem__(self, index):
			original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
			path = self.imgs[index][0]
			tuple_with_path = (original_tuple + (path,))
			return tuple_with_path

	class CropImage(object):
		def __init__(self, top, bottom, left, right):
			self.top = top
			self.bottom = bottom
			self.left = left
			self.right = right

		def __call__(self, img):
			return img.crop((self.left, self.top, img.width - self.right, img.height - self.bottom))


	# Define transformation
	transform = transforms.Compose([
		CropImage(80, 80, 90, 80),  # crop by pixel
		transforms.Resize((224, 224), InterpolationMode.LANCZOS), 
		transforms.ToTensor()
	])

	def load_model(model_path, device, num_classes):    
		model = models.vgg16(pretrained=True)
		for param in model.parameters():
			param.requires_grad = False
		model.avgpool = nn.Identity()
		model.classifier = nn.Sequential(
			nn.Linear(25088, 4096, bias=True), 
			nn.ReLU(inplace=True), 
			nn.Linear(4096, 4096, bias=True),
			nn.ReLU(inplace=True),
			nn.Linear(4096, num_classes, bias=True)
	    	)
		model.load_state_dict(torch.load(model_path, map_location=device))
		model = model.to(device)
		model.eval()
		return model

	def load_model(model_path, device, num_classes):    
		if not os.path.isfile(model_path):
			print(f"The path {model_path} does not point to a valid file. Skipping this model...")
			return None

		model = models.vgg16(pretrained=True)
		for param in model.parameters():
			param.requires_grad = False
		model.avgpool = nn.Identity()
		model.classifier = nn.Sequential(
			nn.Linear(25088, 4096, bias=True), 
			nn.ReLU(inplace=True), 
			nn.Linear(4096, 4096, bias=True),
			nn.ReLU(inplace=True),
			nn.Linear(4096, num_classes, bias=True)
		)

		try:
			model.load_state_dict(torch.load(model_path, map_location=device))
		except Exception as e:
			print(f"Error loading model from {model_path}: {e}")
			return None

		model = model.to(device)
		model.eval()
		return model

	def predict_image_class(image_path, models, device):
		image = Image.open(image_path)
		image = transform(image).unsqueeze(0).to(device)
		outputs = [model(image) for model in models]
		avg_output = torch.stack(outputs).mean(0)
		_, predicted_class = torch.max(avg_output, 1)
		return predicted_class.item()

	def predict_directory(directory_path, model_paths, num_classes):
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		models = [load_model(model_path, device, num_classes) for model_path in model_paths]
		models = [model for model in models if model is not None]  # Remove any None values

		if not models:
			print("No valid models could be loaded.")
			return {}

		image_paths = glob.glob(os.path.join(directory_path, '*.jpg'))  # Get all files in the directory
		class_images = defaultdict(list)  # Dictionary where the keys are class indices and the values are lists of image paths
		for image_path in tqdm(image_paths):
			predicted_class = predict_image_class(image_path, models, device)
			class_images[predicted_class].append(image_path)  # Add the image path to the correct class

		# Save the image paths for each class in separate files
		for class_index, image_paths in class_images.items():
			file_path = os.path.join(directory_path, f'class_{class_index}_images.txt')
			if os.path.exists(file_path):
				print(f"The file {file_path} already exists.")
				response = input("Do you want to continue and overwrite it? (yes/no): ")
				if response.lower() != 'yes':
					print("Skipping this file.")
					continue
			with open(file_path, 'w') as f:
				for path in image_paths:
					f.write(path + '\n')

		return class_images


    
	# paths to your trained models
	# model_paths = [f'./Models/Model_ReduceClasses_Ensemble{i}.pth' for i in [11]]

	# num_classes in your dataset
	num_classes = 3

	predictions = predict_directory(dir_select, model_paths, num_classes)

	print(f"\nPredctions Complete for {model_paths}")

	# You can now print or otherwise use the predictions...
	#for image_path, predicted_class in predictions.items():
	#	print(f"Image: {image_path}, Predicted class: {predicted_class}")



# View images with predictions 
def CNN_select_cut(pred_text):
	import glob

	# Read the file for class 0 and get the image paths
	with open(os.path.join(dir_select, pred_text), 'r') as f:
		class_image_paths = [line.strip() for line in f.readlines()]

	# Load the images into a list
	image_list = []
	for image_path in class_image_paths:
		image = Image.open(image_path)
		photo_image = ImageTk.PhotoImage(image)
		image_list.append((photo_image, image_path))


	newWindow2 = Toplevel(root)
	newWindow2.title('Image Viewer')
	newWindow2.geometry("695x695")

	global my_label
	my_label = Label(newWindow2, image=image_list[0][0])
	my_label.place(x=0, y=0)

	grid_mode = False

	current_image_index = 0  # Add this line to keep track of the current image index

	import re  # Regular expression library

	def go_to_image(event_num):
		global good_events
		event_idx = np.where(good_events == event_num)[0][0]

		for idx, img_path in enumerate(image_list):
			match = re.search(rf'{event_idx}\.jpg$', img_path[1])  # img_path[1] should contain the image file path
			if match:
				update_single(idx)
				break
			
	def change_mode():
		nonlocal grid_mode
		grid_mode = not grid_mode
		if grid_mode:
			button_grid.config(text="Single Image")
			newWindow2.geometry("1025x995")  # Increase window size for 3x3 grid view
			update_grid(current_image_index)  # Pass current_image_index
		else:
			button_grid.config(text="3x3 Grid")
			newWindow2.geometry("695x695")
			update_single(current_image_index)

	def update_single(index):
		nonlocal grid_mode, current_image_index  # Add current_image_index to nonlocal variables
		current_image_index = index  # Update current_image_index with the new index
		if not grid_mode:
			for label in newWindow2.place_slaves():  # Destroy grid labels
				if label != my_label and label not in (button_back, button_forward, button_grid):
    					label.destroy()
			my_label.config(image=image_list[index][0])
			button_forward.config(command=lambda: update_single(index + 1))
			button_back.config(command=lambda: update_single(index - 1))

			if index == 0:
				button_back.config(state=DISABLED)
			else:
				button_back.config(state=NORMAL)

			if index == len(image_list) - 1:
				button_forward.config(state=DISABLED)
			else:
				button_forward.config(state=NORMAL)

			print(f'{index+1} of {len(image_list)}')
			print('Cut Index:', os.path.basename(image_list[index][1]))

			# Set button placement for single image viewer
			button_back.place(x=200, y=10)
			button_forward.place(x=485, y=10)
			button_grid.place(x=330, y=10)

			# Create and place the entry field and "Go to Image" button
			entry_field = Entry(newWindow2, width=10)
			entry_field.place(x=330, y=635)
			go_to_image_button = Button(newWindow2, text="Go to Image", command=lambda: go_to_image(int(entry_field.get())))
			go_to_image_button.place(x=325, y=665)

	def update_grid(index):
		nonlocal grid_mode, current_image_index
		current_image_index = index
		if grid_mode:
			my_label.config(image="")
			for label in newWindow2.place_slaves():
				if label != my_label and label not in (button_back, button_forward, button_grid):
					label.destroy()

		for i in range(3):
			for j in range(3):
				img_idx = index + i * 3 + j
				if img_idx < len(image_list):
					im = image_list[img_idx][1]  # Load original image path
					im = Image.open(im)  # Open the image using PIL
					im.thumbnail((420, 340), Image.Resampling.LANCZOS)  # Now the thumbnail method should work
					grid_image = ImageTk.PhotoImage(im)
					grid_label = Label(newWindow2, image=grid_image)
					grid_label.image = grid_image  # Keep a reference to avoid garbage collection
					grid_label.place(x=320 * j+ 22 * j, y=40+240 * i + 70*i)  

				else:
					break


		button_forward.config(command=lambda: update_grid(min(index + 9, len(image_list) - 1)))
		button_back.config(command=lambda: update_grid(max(index - 9, 0)))

		if index == 0:
			button_back.config(state=DISABLED)
		else:
			button_back.config(state=NORMAL)

		if index + 9 >= len(image_list):
			button_forward.config(state=DISABLED)
		else:
			button_forward.config(state=NORMAL)

		# Set button placement for 3x3 grid view
		button_back.place(x=330, y=10)
		button_forward.place(x=650, y=10)
		button_grid.place(x=470, y=10)



	def forward2(image_number):
		if grid_mode:
			update_grid(image_number)
		else:
			update_single(image_number)

	def back2(image_number):
		if grid_mode:
			update_grid(image_number)
		else:
			update_single(image_number)

	button_back = Button(newWindow2, text="<<", command=lambda: back2(0), state=DISABLED)
	button_forward = Button(newWindow2, text=">>", command=lambda: forward2(1))
	button_grid = Button(newWindow2, text="3x3 Grid", command=change_mode)

	button_back.place(x=200, y=10)
	button_forward.place(x=485, y=10)
	button_grid.place(x=330, y=10)

	update_single(0)

	return





# Gui Code Section *************************************************************************************************************

from tkinter import *
from PIL import ImageTk,Image
import matplotlib.pyplot as plt
import numpy as np
import os
from tkinter import filedialog
import random
import math
from skspatial.objects import Line
import torch
import torch.nn as nn 


# Simple Identity class that let's input pass without changes
class Identity(nn.Module):
	def __init__(self):
		super(Identity, self).__init__()

	def forward(self, x):
		return x

root= Tk()
root.title("GADGET II Analysis Gadget")
#root.iconbitmap("/mnt/projects/e21072/OfflineAnalysis/backups/icons8-physics-48.ico")
if "nt" == os.name:
    root.wm_iconbitmap(bitmap = "/mnt/projects/e21072/OfflineAnalysis/backups/icons8-physics-48.ico")
else:
    root.wm_iconbitmap(bitmap = "@/mnt/projects/e21072/OfflineAnalysis/backups/icons8-physics-48.xbm")

canvas1 = Canvas(root, width=823, height=610, bg="#18453b")
canvas1.pack()

frame = Frame(root, bg="white")
frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)

image = Image.open("/mnt/projects/e21072/OfflineAnalysis/backups/art_GADGETII.PNG")

 
# Resize the image using resize() method
resize_image = image.resize((500, 500))
 
img = ImageTk.PhotoImage(resize_image)

# Create a Label Widget to display the text or Image
label = Label(frame, image = img)
label.place(x=160, y=-10)

label_text = Label(root, text="GADGET II Analysis Gadget",bg="#18453b", fg="white", font = ('times','22'))
label_text.place(x=280, y=10)

label_text2 = Label(root, text="Enter Run # (4 digits)",bg="white", fg="black",font=20)
label_text2.place(x=93, y=63)

label_text3 = Label(root, text="Version 4.23",bg="#18453b", fg="white", font = ('times','12'))
label_text3.place(x=380, y=38)


def get_square_root():  
    x1 = entry1.get()
    
    label1 = Label(root, text=float(x1)**0.5)
    canvas1.create_window(320, 140, window=label1)
    

def create_plot():  
    x = np.linspace(1, 10, 10)
    y = np.linspace(1, 10, 10)
    
    plt.scatter(x, y)
    label2 = plt.show()
    canvas1.create_window(320, 140, window=label2) 
    
def create_image():
    frame = Frame(root, width=10, height=80, bg="white")
    frame.pack()
    frame.place(relwidth=.6, relheight=0.6, relx=0.3, rely=0.17)

    # Create an object of tkinter ImageTk
    #img = ImageTk.PhotoImage(Image.open("Run_4_Indv_Image_2.png"))
    x = np.linspace(1, 10, 10)
    y = np.linspace(1, 10, 10)
    
    fig = plt.scatter(x, y)
    plt.savefig("test_fig.png")
    img = ImageTk.PhotoImage(Image.open("test_fig.png"))

    # Create a Label Widget to display the text or Image
    label = Label(frame, image = img)
    label.pack()
    
    win.mainloop()
   

# Delete entry text    
def temp_text2(e):
    entry_event.delete(0,"end")

def temp_text3(e):
    entry_bins.delete(0,"end")


# Define primary opening functions
def open_eng_spec():
    global run_num
    run_num = entry_run.get()
    #num_bins = int(entry_bins.get())
    energy_spectrum()
    #my_img = ImageTk.PhotoImage(Image.open(plot))
    #my_label = Label(top, image=my_img).pack()

def open_heat_map():
    run_num = entry_run.get()
    plot = energy_spectrum(run_num, num_bins)
    my_img = ImageTk.PhotoImage(Image.open(plot))
    my_label = Label(top, image=my_img).pack()

def open_RvE():
    global run_num
    run_num = entry_run.get()
    RvE()

def open_peak_fit():
    global run_num
    run_num = entry_run.get()
    plot = peak_fit(run_num)
    my_img = ImageTk.PhotoImage(Image.open(plot))
    my_label = Label(top, image=my_img).pack()
    
def open_3d_plot():
    global run_num
    run_num = entry_run.get()
    plot_3D(run_num)
    
def open_track_trace():
    global run_num
    run_num = entry_run.get()
    track_trace(run_num)

def open_track_angles():
    track_angles()

def open1():
    global my_img
    top = Toplevel()
    top.title('My Second Window')
    top.iconbitmap('icons8-physics-48.ico')
    my_img = ImageTk.PhotoImage(Image.open("confidence_level.png"))
    my_label = Label(top, image=my_img).pack()
    btn2 = Button(top, text="close window", command=top.destroy).pack()

def open_start():
    global run_num
    run_num = entry_run.get()
    #num_bins = int(entry_bins.get())
    plot = start(run_num)
    my_img = ImageTk.PhotoImage(Image.open(plot))
    my_label = Label(top, image=my_img).pack()

def open_find():
    global run_num
    run_num = entry_run.get()
    find(run_num)


def open_create_files():
    global run_num
    run_num = entry_run.get()
    plot = create_files(run_num)
    my_img = ImageTk.PhotoImage(Image.open(plot))
    my_label = Label(top, image=my_img).pack()



# Define our switch functions
def switch_peak():
	global is_on_peak

	# Determine is on or off
	if is_on_peak:
		is_on_peak = False
		entry_low.destroy()
		entry_high.destroy()
		button_apply.destroy()		
		
	else:
		is_on_peak = True
		button_peak.configure(fg='green')
		open_peak_fit()

		
def switch_start():
	global is_on_start

	if r1_v.get() == False:
		open_start()	
	else:
		open_find()

def run_num_entry(e):
	button_start["state"] = "normal"    
		
	

# Setting Global Values
is_on_peak = False
is_on_eng_spec = False
is_on_RvE = False
is_on_start = False


# Creating Buttons and Entries **************************************************************************************************

# Run Number Entry
global rand_num
rand_num = random.randrange(0,1000000,1)

entry_run = Entry(root, borderwidth=5) 
entry_run.bind("<FocusIn>", run_num_entry)
canvas1.create_window(163, 97, window=entry_run)

# File Radio Buttons
def my_upd():
    #print('Radiobutton  value :',r1_v.get())
    if not r1_v.get():
        print('Set to File Creation')
        

r1_v = BooleanVar()   # We assigned Boolean variable here
r1_v.set(True) # Can assign False 

r1 = Radiobutton(root, text='Use Existing \n Files', bg='white', highlightthickness = 0, font = ('helvetica','9'), variable=r1_v, value=True,command=my_upd)
canvas1.create_window(125, 130, window=r1) 

r2 = Radiobutton(root, text='Create New \n Files', bg='white', highlightthickness = 0, font = ('helvetica','9'), variable=r1_v, value=False,command=my_upd)
canvas1.create_window(200, 130, window=r2)

# Start Button    
button_start = Button(text='START', borderwidth=5, fg='green', command=switch_start)
button_start["state"] = "disabled"
canvas1.create_window(163, 170, window=button_start)

# Energy Spectrum Button    
button_eng_spec = Button(text='Energy Spectrum',fg='green', command=open_eng_spec)
canvas1.create_window(163, 235, window=button_eng_spec)
button_eng_spec["state"] = "disabled"

# Range vs Energy Button
button_RvE = Button(text='Range vs Energy',fg='green',command=open_RvE)
canvas1.create_window(163, 285, window=button_RvE)
button_RvE["state"] = "disabled"

# 3D Plot Button
button_3D = Button(text='3D Event Plot',fg='green', command=open_3d_plot)
canvas1.create_window(163, 335, window=button_3D)
button_3D["state"] = "disabled"

# Track with Trace Button
button_track_trace = Button(text='Track with Trace',fg='green', command=open_track_trace)
canvas1.create_window(163, 385, window=button_track_trace)
button_track_trace["state"] = "disabled"

# Track Angle Button
button_track_angle = Button(text='Track Angles',fg='green', command=open_track_angles)
canvas1.create_window(163, 435, window=button_track_angle)
button_track_angle["state"] = "disabled"

# ConvNet Button
button_cnn = Button(text='ConvNet Track ID',fg='green', command=cnn)
canvas1.create_window(163, 485, window=button_cnn)
button_cnn["state"] = "disabled"

root.mainloop()


