import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

print(current, parent)


from my_module import hello

hello()




# getting the name of the directory
# where the this file is present.

# Getting the parent directory name
# where the current directory is present.

# adding the parent directory to 
# the sys.path.

# now we can import the module in the parent
# directory.
