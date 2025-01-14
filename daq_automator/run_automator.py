import datetime
import time
from apscheduler.schedulers.background import BackgroundScheduler
from pynput import mouse
from pynput.mouse import Button, Controller

# Choose stop and start locations
x_stop = 1895
y_stop = 10
x_start = 1895
y_start = 10

# Start the scheduler
sched = BackgroundScheduler()
sched.daemonic = False

# Control of the auto clicker
mouse_controller = Controller() 

def job_function_xia():
    f.open("run_scheduler_notes_xia.txt","a")
    time.sleep(10) # wait 10 seconds after GET stops
    mouse_controller.position = (x_stop,y_stop) # move mouse to stop button
    mouse_controller.press(Button.left)
    mouse_controller.release(Button.left)
    f.write("XIA DAQ Stopped: ",datetime.datetime.now())
    time.sleep(5) # wait for the DAQ to properly reset (3 seconds)
    mouse_controller.position = (x_start,y_start) # move mouse to start button
    mouse_controller.press(Button.left)
    mouse_controller.release(Button.left)
    f.write("XIA DAQ Started: ",datetime.datetime.now())
    f.close()

def job_function_get():
    f.open("run_scheduler_notes_get.txt","a")
    mouse_controller.position = (x_stop,y_stop) # move mouse to stop button
    mouse_controller.press(Button.left)
    mouse_controller.release(Button.left)
    f.write("GET DAQ Stopped: ",datetime.datetime.now())
    time.sleep(17) # wait for xia daq to stop and start again, then begin GET DAQ 2 seconds after XIA begins
    mouse_controller.position = (x_start,y_start) # move mouse to start button
    mouse_controller.press(Button.left)
    mouse_controller.release(Button.left)
    f.write("GET DAQ Started: ",datetime.datetime.now())
    f.close()

# Schedules job_function to be run once every 4 hours, starting from the minimum
sched.add_job(job_function, trigger='cron',  hour='*/4')
sched.start()