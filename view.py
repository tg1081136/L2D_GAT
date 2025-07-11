import ast
import matplotlib.pyplot as plt
import pandas as pd

# problem params
n_j = 3
n_m = 3
l = 1
h = 99
stride = 50
datatype = 'log'  # 'vali', 'log'

with open('./{}_{}_{}_{}_{}.txt'.format(datatype, n_j, n_m, l, h), 'r') as file:
    content = file.read()
    data = ast.literal_eval(content)  # 安全地把文字轉換為 Python list


# 分離 x 和 y
steps = [point[0] for point in data]
logs = [point[1] for point in data]
losses = [point[2] for point in data]
smooth_logs = pd.Series(logs).rolling(window=10).mean()
# 畫圖
plt.figure(figsize=(10, 5))
plt.plot(steps, smooth_logs, linestyle='-', color='tomato', label='Smoothed Reward')
#plt.plot(steps, losses, linestyle='-', color='blue', label='Smoothed Reward')
plt.title("JSSP Training Log Over Time")
plt.xlabel("Step")
plt.ylabel("Log")
#plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.show()