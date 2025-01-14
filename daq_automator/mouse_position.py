from pynput import mouse
from pynput.mouse import Button

mouse_controller = mouse.Controller()
print(mouse_controller.position)

