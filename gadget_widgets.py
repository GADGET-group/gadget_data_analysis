from tkinter import ttk
import tkinter as tk

class GEntry(ttk.Entry):
    '''
    Same as tkinter Entry, but clears and restores default text on focus change if
    the field is empty.
    '''
    def __init__(self, master=None, default_text='', **kw):
        super().__init__(master, **kw)
        self.bind('<FocusIn>', self.on_focus_in)
        self.bind('<FocusOut>', self.on_focus_out)
        self.default_text = default_text
        self.insert(0, default_text)

    def on_focus_in(self, event):
        if event.widget.get() == event.widget.default_text:
            event.widget.delete(0, tk.END)

    def on_focus_out(self, event):
        if event.widget.get() == '':
            event.widget.insert(0, event.widget.default_text)

from PIL import Image, ImageTk
def get_background_image():
        return None
        return ImageTk.PhotoImage(Image.open(
            '/mnt/projects/e21072/OfflineAnalysis/backups/art_GADGETII.PNG').resize((500,500)))