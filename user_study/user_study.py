import pandas as pd
import numpy as np

df = pd.read_excel("./20240514.xlsx")

# print(df)

user_cnt = 0
ours_emo = 0
ours_stru = 0
bd_emo = 0
bd_stru = 0
p2p0_emo = 0
p2p0_stru = 0
sdedit_emo = 0
sdedit_stru = 0

emo_index = range(5, 85, 2)
stru_index = range(6, 86, 2)

# print(len(emo_index))
# print(len(stru_index))
ours_emo_list = []
bd_emo_list = []
p2p0_emo_list = []
sdedit_emo_list = []
ours_stru_list = []
bd_stru_list = []
p2p0_stru_list = []
sdedit_stru_list = []


for index, row in df.iterrows():
    t_ours_emo = 0
    t_bd_emo = 0
    t_p2p0_emo = 0
    t_sdedit_emo = 0
    for i in emo_index:
        temp = row[i]
        if "A" in temp:
            ours_emo += 1
            t_ours_emo += 1
        elif "B" in temp:
            bd_emo += 1
            t_bd_emo += 1
        elif "C" in temp:
            p2p0_emo += 1
            t_p2p0_emo += 1
        elif "D" in temp:
            sdedit_emo += 1
            t_sdedit_emo += 1
        else:
            print("error")
    t_ours_stru = 0
    t_bd_stru = 0
    t_p2p0_stru = 0
    t_sdedit_stru = 0
    for i in stru_index:
        temp = row[i]
        if "A" in temp:
            ours_stru += 1
            t_ours_stru += 1
        elif "B" in temp:
            bd_stru += 1
            t_bd_stru += 1
        elif "C" in temp:
            p2p0_stru += 1
            t_p2p0_stru += 1
        elif "D" in temp:
            sdedit_stru += 1
            t_sdedit_stru += 1
        else:
            print("error")
    user_cnt += 1
    ours_emo_list.append(t_ours_emo / 40)
    bd_emo_list.append(t_bd_emo / 40)
    p2p0_emo_list.append(t_p2p0_emo / 40)
    sdedit_emo_list.append(t_sdedit_emo / 40)
    ours_stru_list.append(t_ours_stru / 40)
    bd_stru_list.append(t_bd_stru / 40)
    p2p0_stru_list.append(t_p2p0_stru / 40)
    sdedit_stru_list.append(t_sdedit_stru / 40)
# if user_cnt > 20:
#     break
print(user_cnt)
print("emotion: ")
print(
    "ours: ",
    ours_emo,
    ours_emo / (40 * user_cnt),
    np.std(ours_emo_list),
    np.percentile(ours_emo_list, 25),
    np.percentile(ours_emo_list, 75),
)
print(
    "bd: ",
    bd_emo,
    bd_emo / (40 * user_cnt),
    np.std(bd_emo_list),
    np.percentile(bd_emo_list, 25),
    np.percentile(bd_emo_list, 75),
)
print(
    "p2p0: ",
    p2p0_emo,
    p2p0_emo / (40 * user_cnt),
    np.std(p2p0_emo_list),
    np.percentile(p2p0_emo_list, 25),
    np.percentile(p2p0_emo_list, 75),
)
print(
    "sdedit: ",
    sdedit_emo,
    sdedit_emo / (40 * user_cnt),
    np.std(sdedit_emo_list),
    np.percentile(sdedit_emo_list, 25),
    np.percentile(sdedit_emo_list, 75),
)
print()
print("structure:")
print(
    "ours: ",
    ours_stru,
    ours_stru / (40 * user_cnt),
    np.std(ours_stru_list),
    np.percentile(ours_stru_list, 25),
    np.percentile(ours_stru_list, 75),
)
print(
    "bd: ",
    bd_stru,
    bd_stru / (40 * user_cnt),
    np.std(bd_stru_list),
    np.percentile(bd_stru_list, 25),
    np.percentile(bd_stru_list, 75),
)
print(
    "p2p0: ",
    p2p0_stru,
    p2p0_stru / (40 * user_cnt),
    np.std(p2p0_stru_list),
    np.percentile(p2p0_stru_list, 25),
    np.percentile(p2p0_stru_list, 75),
)
print(
    "sdedit: ",
    sdedit_stru,
    sdedit_stru / (40 * user_cnt),
    np.std(sdedit_stru_list),
    np.percentile(sdedit_stru_list, 25),
    np.percentile(sdedit_stru_list, 75),
)
print()

ours_cnt = 0
bd_cnt = 0
p2p0_cnt = 0
sdedit_cnt = 0
youxiao_cnt = 0
total_cnt = 0

ours_youxiao_list = []
bd_youxiao_list = []
p2p0_youxiao_list = []
sdedit_youxiao_list = []

for index, row in df.iterrows():
    ours_youxiao_cnt = 0
    bd_youxiao_cnt = 0
    p2p0_youxiao_cnt = 0
    sdedit_youxiao_cnt = 0
    for i in emo_index:
        if "A" in row[i] and "A" in row[i + 1]:
            ours_cnt += 1
            youxiao_cnt += 1
            ours_youxiao_cnt += 1
        elif "B" in row[i] and "B" in row[i + 1]:
            bd_cnt += 1
            youxiao_cnt += 1
            bd_youxiao_cnt += 1
        elif "C" in row[i] and "C" in row[i + 1]:
            p2p0_cnt += 1
            youxiao_cnt += 1
            p2p0_youxiao_cnt += 1
        elif "D" in row[i] and "D" in row[i + 1]:
            sdedit_cnt += 1
            youxiao_cnt += 1
            sdedit_youxiao_cnt += 1
        # else:
        #     print("error")
        total_cnt += 1
    total_youxiao = (
        ours_youxiao_cnt + bd_youxiao_cnt + p2p0_youxiao_cnt + sdedit_youxiao_cnt
    )
    ours_youxiao_list.append(ours_youxiao_cnt / total_youxiao)
    bd_youxiao_list.append(bd_youxiao_cnt / total_youxiao)
    p2p0_youxiao_list.append(p2p0_youxiao_cnt / total_youxiao)
    sdedit_youxiao_list.append(sdedit_youxiao_cnt / total_youxiao)
print(youxiao_cnt, total_cnt)
print(
    "ours: ",
    ours_cnt,
    ours_cnt / youxiao_cnt,
    np.std(ours_youxiao_list),
    np.percentile(ours_youxiao_list, 25),
    np.percentile(ours_youxiao_list, 75),
)
print(
    "bd: ",
    bd_cnt,
    bd_cnt / youxiao_cnt,
    np.std(bd_youxiao_list),
    np.percentile(bd_youxiao_list, 25),
    np.percentile(bd_youxiao_list, 75),
)
print(
    "p2p0: ",
    p2p0_cnt,
    p2p0_cnt / youxiao_cnt,
    np.std(p2p0_youxiao_list),
    np.percentile(p2p0_youxiao_list, 25),
    np.percentile(p2p0_youxiao_list, 75),
)
print(
    "sdedit: ",
    sdedit_cnt,
    sdedit_cnt / youxiao_cnt,
    np.std(sdedit_youxiao_list),
    np.percentile(sdedit_youxiao_list, 25),
    np.percentile(sdedit_youxiao_list, 75),
)
