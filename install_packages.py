# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1  2023

script to install needed packages

@author: Alex Kedziora
"""

import sys
import subprocess

# implement pip as a subprocess:
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyvisa'])
#subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'serial'])
