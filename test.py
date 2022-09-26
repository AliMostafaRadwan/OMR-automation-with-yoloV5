import pyautogui, sys
print('Press Ctrl-C to quit.')
try:
    i = 0
    while True:
        i += 1
        print(i, end='\r', flush=True)
        
except KeyboardInterrupt:
    print('\n')