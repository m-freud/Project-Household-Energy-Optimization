import os

current_dir = os.path.dirname(os.path.abspath(__file__))
print("Current directory:", current_dir)

print(os.path.join(current_dir, "..", "db"))

print(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "db")))


print(__file__)

print(os.getcwd())