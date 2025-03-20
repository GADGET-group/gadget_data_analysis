import sys

import tkinter as tk

from raw_viewer.RawEventViewerFrame import RawEventViewerFrame

if __name__ == '__main__':
    root = tk.Tk()
    RawEventViewerFrame(root, heritage_file='-old' in sys.argv).grid()
    root.mainloop()