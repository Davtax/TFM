import time

import numpy as np
from tqdm import tqdm
import sys

pbar = tqdm(total=100, desc='Processing', file=sys.stdout, ncols=90, bar_format='{l_bar}{bar}{r_bar}')

for i in range(0, 100):
	time.sleep(np.random.random() * 0.1)
	pbar.update()
