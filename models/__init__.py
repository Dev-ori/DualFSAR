import os
import importlib

# Get the directory of the current file (__init__.py)
directory = os.path.dirname(__file__)

# Loop over the files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.py') and filename != '__init__.py':
        # Remove the .py extension and import the module
        module_name = filename[:-3]
        importlib.import_module('.' + module_name, package=__name__)