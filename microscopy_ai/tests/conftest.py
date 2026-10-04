import os
import sys

HERE = os.path.dirname(__file__)
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "..", "openjev"), os.path.join(HERE, "..", "..", "openjev", "tests")]
