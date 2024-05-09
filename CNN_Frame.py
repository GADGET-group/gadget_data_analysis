import datetime
import random
import os
import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
from matplotlib.path import Path
from GadgetRunH5 import GadgetRunH5
import numpy as np
from tqdm import tqdm
import gadget_widgets
from prev_cut_select_window import PrevCutSelectWindow
import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
import glob
from collections import defaultdict
from PIL import Image, ImageTk
import numpy as np
import os
import glob
import re
import csv
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import easyocr

class CNN_Frame(ttk.Frame):
    def __init__(self, parent, run_data: GadgetRunH5):
        super().__init__(parent)
        self.run_data = run_data
        self.background_image = gadget_widgets.get_background_image()
        self.background = ttk.Label(self, image=self.background_image)
        self.background.place(relx=0.5, rely=0.5, anchor='center')
        self.models = []
        self.prediction_buttons = []
        # UI setup
        self.cut_tools_frame = ttk.LabelFrame(self, text='Select Data')
        self.cut_tools_frame.grid(row=2)
        self.manual_cut_button = ttk.Button(self.cut_tools_frame, text='Manual Cut Selection')
        self.manual_cut_button.grid(row=0, column=0)
        self.from_file_cut_button = ttk.Button(self.cut_tools_frame, text='Polygon from File', command=self.cut_from_file)
        self.from_file_cut_button.grid(row=0, column=1)
        self.prev_cut_button = ttk.Button(self.cut_tools_frame, text='Previous Cuts', command=self.prev_cut)
        self.prev_cut_button.grid(row=1, column=0, columnspan=2)

        self.model_select_frame = ttk.LabelFrame(self, text='Select & Deploy')
        self.model_select_frame.grid(row=4)
        self.model_select_button = ttk.Button(self.model_select_frame, text='Select Trained CNN Model(s)', command=self.select_model)
        self.model_select_button.grid(row=1, column=0)
        self.deploy_button = ttk.Button(self.model_select_frame, text='Deploy Model(s)', command=self.predict)
        self.deploy_button.grid(row=2, column=0)

    def load_prediction_files(self):
        prediction_dir = self.glob_dir_select  # Adjust as necessary
        txt_files = glob.glob(os.path.join(prediction_dir, '*.txt'))
        for idx, file_path in enumerate(txt_files):
            self.create_prediction_button(file_path, idx)


    def create_prediction_button(self, file_path, idx):
        file_name = os.path.basename(file_path)
        event_type = file_name.replace('.txt', '').replace('_', ' ').capitalize()
        btn = ttk.Button(self, text=f"View {event_type}", command=lambda f=file_path: self.CNN_select_cut(f))
        btn.grid(row=idx+5, column=0, sticky='ew')  # Use grid with dynamic row assignment
        self.prediction_buttons.append(btn)

    def cut_from_file(self):
        fname = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select file")
        if fname:  # Check if a file was actually selected
            points = np.loadtxt(fname)
            self.save_cut_files(points)

    def save_cut_files(self, points):
        now = datetime.datetime.now()
        rand_num = str(random.randrange(0, 1000000))
        cut_name = f"{rand_num}_CUT_Date_{now.strftime('%m_%d_%Y')}"
        event_images_path = os.path.join(self.run_data.folder_path, cut_name)
        os.makedirs(event_images_path, exist_ok=True)  # Ensure directory exists without raising an error if it already exists

        self.plot_spectrum(fig_name=cut_name)  # Assuming you have already defined plot_spectrum elsewhere in your class
        ax = plt.gca()
        
        points_list = list(points) + [[points[0][0], points[0][1]]]  # Close the polygon by adding the first point at the end
        codes = [Path.MOVETO] + [Path.LINETO] * (len(points_list) - 1) + [Path.CLOSEPOLY]
        path = Path(points_list, codes)
        patch = patches.PathPatch(path, fill=False, color='red', lw=2)
        ax.add_patch(patch)
        plt.savefig(os.path.join(event_images_path, f'{cut_name}.jpg'))
        plt.close()

        # Saving the selected indices and images for each event
        selected_indices = self.run_data.get_RvE_cut_indexes(points)
        for index in tqdm(selected_indices, desc="Saving selected images"):
            image_name = f"run{self.run_data.run_num}_image_{index}.jpg"
            image_path = os.path.join(event_images_path, image_name)
            self.run_data.save_image(index, save_path=image_path)  # Assuming you have a method save_image in GadgetRunH5

        # Save the cut parameters used
        np.savetxt(os.path.join(event_images_path, 'cut_used.txt'), points, fmt='%f')



    def select_model(self):
        mypath = "/mnt/analysis/e21072/models"
        selected_files = filedialog.askopenfilenames(initialdir=mypath, title="Select Model(s)")
        self.models.extend(selected_files)

    def predict(self):
        if not self.models:
            print("No models selected.")
            return
        num_classes = 3
        predictions = self.predict_directory(self.glob_dir_select, self.models, num_classes)
        print(f"\nPredictions Complete for {self.models}")

    def predict_directory(self, directory_path, model_paths, num_classes):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        models = [self.load_model(path, device, num_classes) for path in model_paths]
        models = [model for model in models if model is not None]  # Remove any None values

        if not models:
            print("No valid models could be loaded.")
            return {}

        image_paths = glob.glob(os.path.join(directory_path, '*.png'))  # Get all files in the directory
        class_images = defaultdict(list)  # Dictionary where the keys are class indices and the values are lists of image paths
        for image_path in tqdm(image_paths):
            predicted_class = self.predict_image_class(image_path, models, device)
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


    def load_model(self, model_path, device, num_classes):
        try:
            model = models.vgg16(pretrained=True)
            model.avgpool = nn.Identity()
            model.classifier = nn.Sequential(
                nn.Linear(25088, 4096), nn.ReLU(inplace=True),
                nn.Linear(4096, 4096), nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes)
            )
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            return model
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            return None

    def predict_image_class(self, image_path, models, device):
        transform = transforms.Compose([transforms.ToTensor()])
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        outputs = [model(image) for model in models]
        avg_output = torch.stack(outputs).mean(0)
        _, predicted_class = torch.max(avg_output, 1)
        return predicted_class.item()

    def prev_cut(self):
        window = PrevCutSelectWindow(self, self.run_data)
        self.wait_window(window)  # This will block until the window is closed

        if hasattr(window, 'image_path_list') and len(window.image_path_list) > window.current_image_index:
            self.glob_dir_select = window.image_path_list[window.current_image_index][:-4]
        
        self.load_prediction_files()
       
    def CNN_select_cut(self, pred_text):
        print(f'\nLoading predictions from {pred_text}')
        dir_select = os.path.dirname(pred_text)  # Get the directory of the file
        self.current_image_index = 0

        with open(pred_text, 'r') as f:
            class_image_paths = [line.strip() for line in f.readlines()]

        self.image_list = class_image_paths
        # for image_path in class_image_paths:
        #     image = Image.open(image_path)
        #     photo_image = ImageTk.PhotoImage(image)
        #     self.image_list.append((photo_image, image_path))

        self.create_image_viewer()

    def create_image_viewer(self):
        self.newWindow = tk.Toplevel(self)
        self.newWindow.title('Image Viewer')
        self.newWindow.geometry("900x680")

        self.my_label = tk.Label(self.newWindow, image=None)
        self.my_label.grid(row=0, column=0, padx=10, pady=10)

        self.filename_label = tk.Label(self.newWindow, text="", font=('Helvetica', 14))
        self.filename_label.grid(row=1, column=0)

        self.button_back = tk.Button(self.newWindow, text="<<", command=lambda: self.navigate_images(-1))
        self.button_forward = tk.Button(self.newWindow, text=">>", command=lambda: self.navigate_images(1))

        self.button_back.grid(row=2, column=0, sticky=tk.W, padx=10)
        self.button_forward.grid(row=2, column=2, sticky=tk.E, padx=10)

        # Entry and Go-To Button for specific image navigation
        self.go_to_entry = tk.Entry(self.newWindow)
        self.go_to_entry.grid(row=2, column=1)
        self.go_to_button = tk.Button(self.newWindow, text="Go to Image", command=self.go_to_image)
        self.go_to_button.grid(row=3, column=1, pady=10)
        self.current_image_index = 0
        self.update_image_display(self.current_image_index)

    def navigate_images(self, direction):
        new_index = max(0, min(self.current_image_index + direction, len(self.image_list) - 1))
        self.update_image_display(new_index)

    def go_to_image(self):
        
        try:
            # Get the user input from the entry box
            image_number = int(self.go_to_entry.get())
            # Construct a pattern to match filenames. Adjust the pattern based on your file naming conventions.
            pattern = rf"image_{image_number}\.png$"

            # Search through the image list to find a matching file
            for index, path in enumerate(self.image_list):
                if re.search(pattern, os.path.basename(path)):
                    self.update_image_display(index)
                    return

            # If no match is found, display a warning message
            tk.messagebox.showwarning('Image Not Found', f'No image found for number: {image_number}')
        except ValueError:
            # Handle cases where the user input is not an integer
            tk.messagebox.showerror('Invalid Input', 'Please enter a valid image number.')


    def update_image_display(self, index):
        self.current_image_index = index
        image_path = self.image_list[index]
        image = Image.open(image_path)

        # Resize image maintaining aspect ratio
        single_image_size = (840, 680)
        aspect_ratio = min(single_image_size[0] / image.width, single_image_size[1] / image.height)
        new_size = (int(image.width * aspect_ratio), int(image.height * aspect_ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

        photo_image = ImageTk.PhotoImage(image)
        self.my_label.config(image=photo_image)
        self.my_label.image = photo_image  # keep a reference!
        
        # Update filename label
        self.filename_label.config(text=f"Image: {os.path.basename(image_path)}")

        # Update button states based on the current index
        self.button_back['state'] = 'normal' if index > 0 else 'disabled'
        self.button_forward['state'] = 'normal' if index < len(self.image_list) - 1 else 'disabled'