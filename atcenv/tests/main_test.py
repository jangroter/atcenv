from jsonargparse import CLI
from test import Dataset
from abc import ABC
import sys

default = Dataset()

def some_function(a,b):
    print(a+b)

def main(Bla : Dataset = default, a = 10, b = 20):
    print(Bla)
    some_function(a,b)
    return

if __name__ == "__main__":
    if '--config' in sys.argv:
        CLI(main)
    else:
        CLI(main, args=['--config', 'config.yaml'])