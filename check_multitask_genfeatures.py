# Print the GenFeatures class from multitask_7dataset.py so we can copy it exactly
import inspect, sys
sys.path.insert(0, '.')
# Just read the file and extract the GenFeatures class
with open('multitask_7dataset.py', 'r') as f:
    lines = f.readlines()

in_class = False
for i, line in enumerate(lines):
    if 'class GenFeatures' in line:
        in_class = True
    if in_class:
        print(f"{i+1}: {line}", end='')
    if in_class and i > 0 and line.startswith('class ') and 'GenFeatures' not in line:
        break
