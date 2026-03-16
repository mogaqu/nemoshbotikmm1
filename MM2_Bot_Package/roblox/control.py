# abstraction API for Roblex keyboard and mouse control

import numpy as np
import random
import time
from ahk import AHK

# key mappings
key = {'up': 'w',
       'down': 's',
       'left': 'a',
       'right': 'd',
       'jump': 'space',
       'turn_left': 'left',
       'turn_right': 'right',
       'look_up': 'up',
       'look_down': 'down'}
keyTurns = [key['turn_left'], key['turn_right']]
ahk = AHK(executable_path='C:/Program Files/AutoHotkey/v2/AutoHotkey.exe')

class Control:
    def __init__(self, simulationMode=False):
        self.actionPressed = []
        self.simulationMode = simulationMode

    def press(self, action):
        '''press the key associated with action '''
        assert action in key, 'Error invalid action %s'%action
        self.actionPressed.append(action)
        if not self.simulationMode:
            ahk.key_down(key[action], blocking=False)

    def release(self, action):
        '''release a specific action if it is pressed'''
        if action in self.actionPressed:
            if not self.simulationMode:
                ahk.key_up(key[action], blocking=False)
            try:
                self.actionPressed.remove(action)
            except ValueError:
                pass

    def release_all_keys(self):
        '''Release all keys pressed'''
        if not self.simulationMode:
            for a in self.actionPressed:
                ahk.key_up(key[a], blocking=False)
        self.actionPressed = []

    def all_move_actions(self):
        ''' return all move actions '''
        return ['up', 'down', 'left', 'right']

    def current_actions(self):
        '''return current actions in motion'''
        return self.actionPressed.copy()

    def press_for(self, action, duration):
        """Presses a key for a given duration."""
        self.press(action)
        time.sleep(duration)
        self.release(action)






