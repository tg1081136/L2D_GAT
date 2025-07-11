import numpy as np

# 載入檔案
data = np.load('C:/Users/ysann/Desktop/專題/L2D_GAT/DataGen/generatedData2_3_Seed200.npy', allow_pickle=True)

# 看一下資料筆數
print(f"總筆數：{len(data)}")

# 看第一筆資料的內容長什麼樣（通常是 tuple）
print("第一筆資料：")
print(data[0])