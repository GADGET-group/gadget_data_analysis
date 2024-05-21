import tkinter as tk

from raw_viewer.RawEventViewerFrame import RawEventViewerFrame

if __name__ == '__main__':
    root = tk.Tk()
    RawEventViewerFrame(root).grid()
    root.mainloop() 