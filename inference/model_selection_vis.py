""" モデル選択可視化 """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import japanize_matplotlib
sys.path.append('../')

ply = [
    1228.37512484,
    1169.55971186,
    1156.87847122,
    1078.44615721,
    1060.4407051,
    1102.28365998,
    1083.04223881
]

plt.plot(range(len(ply)), ply, linewidth=2, color="#666666", marker="x", markersize=10)
plt.xticks(range(len(ply)), [5, 10, 15, 20, 25, 30, 35], fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel("トピック数", fontsize=15)
plt.ylabel("Perplexity", fontsize=15)
plt.show()