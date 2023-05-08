import math
import random
import numpy as np
#################################初始化###############################################
num = 20
delta = 0.9
alpha = 1
gammai = []
Throughput = []
realtuntuliang_jin_1 = []
realtuntuliang_max_1 = []
realtuntuliang_jin_2 = []
realtuntuliang_max_2 = []
realtuntuliang_jin_3 = []
realtuntuliang_max_3 = []

miyou_jin_1 = []
miyou_max_1 = []
miyou_jin_2 = []
miyou_max_2 = []
miyou_jin_3 = []
miyou_max_3 = []

AOI_jin_1 = []
AOI_max_1 = []
AOI_jin_2 = []
AOI_max_2 = []
AOI_jin_3 = []
AOI_max_3 = []

taltolAOI_jin_1 = []
taltolAOI_max_1 = []
taltolAOI_jin_2 = []
taltolAOI_max_2 = []
taltolAOI_jin_3 = []
taltolAOI_max_3 = []

all_Nodes_jin_1 = []
all_Nodes_max_1 = []
all_Nodes_jin_2 = []
all_Nodes_max_2 = []
all_Nodes_jin_3 = []
all_Nodes_max_3 = []

left_Nodes_jin_1 = []
left_Nodes_max_1 = []
left_Nodes_jin_2 = []
left_Nodes_max_2 = []
left_Nodes_jin_3 = []
left_Nodes_max_3 = []

choose_Nodes_jin_1 = [] # 被选择的
choose_Nodes_max_1 = [] # 被选择的

S_jin_1 = 0
S_max_1 = 0
Commurounds_jin_1 = 0
Commurounds_max_1 = 0
Commurounds_jin_2 = 0
Commurounds_max_2 = 0
Commurounds_jin_3 = 0
Commurounds_max_3 = 0

rounds= 100000

possi = []
for i in range(num):
    r1 = random.randint(1,9)/10
    possi.append(r1)

for i in range(len(possi)):
    gammai.append(0)
    Throughput.append(0)
    realtuntuliang_jin_1.append(0)
    realtuntuliang_max_1.append(0)
    realtuntuliang_jin_2.append(0)
    realtuntuliang_max_2.append(0)
    realtuntuliang_jin_3.append(0)
    realtuntuliang_max_3.append(0)
    miyou_jin_1.append(0)
    miyou_max_1.append(0)
    miyou_jin_2.append(2)
    miyou_max_2.append(0)
    miyou_jin_3.append(2)
    miyou_max_3.append(0)
    AOI_jin_1.append(1)
    AOI_max_1.append(1)
    AOI_jin_2.append(1)
    AOI_max_2.append(1)
    AOI_jin_3.append(1)
    AOI_max_3.append(1)
    taltolAOI_jin_1.append(0)
    taltolAOI_max_1.append(0)
    taltolAOI_jin_2.append(0)
    taltolAOI_max_2.append(0)
    taltolAOI_jin_3.append(0)
    taltolAOI_max_3.append(0)
    all_Nodes_jin_1.append(i)
    all_Nodes_max_1.append(i)
    all_Nodes_jin_2.append(i)
    all_Nodes_max_2.append(i)
    all_Nodes_jin_3.append(i)
    all_Nodes_max_3.append(i)

matrix = np.random.randint(1, 10, size=(num, num))
matrix = np.triu(matrix)
matrix += matrix.T - np.diag(matrix.diagonal())

for i in range(num):
    matrix[i][i]=100

matrix = matrix/100
matrix_inv = np.linalg.inv(matrix)
matrixT_inv = np.linalg.inv(matrix.T)
newmatrix = np.array(list(matrix.T))
for i in range(num):
    for j in range(num):
        newmatrix[j][i]=matrix.T[j][i]*possi[i]
newmatrixT_inv=np.linalg.inv(newmatrix)

lambda1_jin_2=1
lambda1_jin_3=1

#################################1静态随机策略###############################################
print("1静态随机策略start")
for i in range(len(possi)):
    Throughput[i] = (delta * possi[i]) / num
    #print(Throughput[i])
for i in range(len(possi)):
    gammai[i] = (alpha * possi[i]) / (num * Throughput[i] * Throughput[i])
    #print(gammai[i])

gamma = max(gammai)
#print(gamma)

for i in range(len(possi)):
    miyou_jin_1[i] = (Throughput[i] / possi[i]) * max(1.0, math.sqrt(gammai[i] / gamma))
 #   print(miyou_i[i])

for i in miyou_jin_1:
    S_jin_1 += i
#print(S)
while S_jin_1 < 1:
    S_jin_1 = 0
    gamma -= 0.1
    for i in range(len(possi)):
        miyou_jin_1[i] = (Throughput[i] / possi[i]) * max(1.0, math.sqrt(gammai[i] / gamma))
 #       print(miyou_i[i])

    for i in miyou_jin_1:
        S_jin_1 += i

#################################2静态随机策略###############################################
print("2静态随机策略start")
while sum(miyou_jin_2) >1:
    print("===================2静态随机策略round",lambda1_jin_2,"===================")
    miyou_jin_2 = []
    for i in range(num):
        miyou_jin_2.append(0)
    lambda1_jin_2 +=50
    for i in range(num):
        #print("------------------",i,"-------------------------")
        for j in range(num):
            #print(matrixT_inv[i][j])
            column_sum = np.sum(matrixT_inv[:, j], axis=0)
            #print("第",j,"列的和为：",column_sum)
            sum1 = newmatrixT_inv[i][j]/ math.sqrt(column_sum)
            #print(sum1)
            miyou_jin_2[i]+=sum1
        miyou_jin_2[i] /= math.sqrt(lambda1_jin_2)
        #print("mu[i]",mu[i])
#################################3静态随机策略###############################################
print("3静态随机策略start")
while sum(miyou_jin_3) >1:
    print("===================3静态随机策略round",lambda1_jin_3,"===================")
    miyou_jin_3 = []
    for i in range(num):
        miyou_jin_3.append(0)
    lambda1_jin_3 +=200
    for i in range(num):
        #print("------------------",i,"-------------------------")
        for j in range(num):
            #print(matrixT_inv[i][j])
            column_sum = np.sum(matrixT_inv[:, j], axis=0)
            #print("第",j,"列的和为：",column_sum)
            sum1 = newmatrixT_inv[i][j]/ math.sqrt(column_sum)
            #print(sum1)
            miyou_jin_3[i]+=sum1
        miyou_jin_3[i] /= math.sqrt(lambda1_jin_3)
        if (Throughput[i] / possi[i]) >= miyou_jin_3[i]:
            miyou_jin_3[i]=Throughput[i] / possi[i]
        #print("mu[i]",mu[i])
#################################加权采样###############################################
def weighted_random_sampling(items, weights):
    # 计算每个元素被选中的概率
    total_weight = sum(weights)
    probabilities = [w / total_weight for w in weights]

    # 使用随机数生成器进行采样
    r = random.random()
    #print(r)
    cumulative_probability = 0
    for item, probability in zip(items, probabilities):
        cumulative_probability += probability
        if r < cumulative_probability:
            #print(item)
            return item
    # 如果未能选择任何元素，则返回 None
    return None

#################################1静态随机仿真###############################################
while Commurounds_jin_1 < rounds:
    print("--------------", "1静态随机仿真rounds", Commurounds_jin_1,"--------------")
    Commurounds_jin_1 += 1
    left_Nodes_jin_1 = []
    #choose_Nodes_jin_1 = 0  # 被选择的
    # 成功调度
    choose_node_jin_1 = weighted_random_sampling(all_Nodes_jin_1, miyou_jin_1)
    ran = random.random()
    #print(ran)
    # 被调度的
    if ran <= possi[choose_node_jin_1]:# 通信成功
            print("yes")
            AOI_jin_1[choose_node_jin_1] = 1
            realtuntuliang_jin_1[choose_node_jin_1]+=1
    else: #失败
            print("no")
            AOI_jin_1[choose_node_jin_1] += 1
    # 没被调度的
    left_Nodes_jin_1 = all_Nodes_jin_1.copy()
    left_Nodes_jin_1.remove(choose_node_jin_1)
    #print("left1",left_Nodes)
    for i in range(len(left_Nodes_jin_1)):
        AOI_jin_1[left_Nodes_jin_1[i]] += 1

    taltolAOI_jin_1 = [taltolAOI_jin_1[i] + AOI_jin_1[i] for i in range(len(AOI_jin_1))]
    print("aoi_jin_1", AOI_jin_1)
    print("taltolaoi_jin_1", taltolAOI_jin_1)
    print("吞吐量_jin_1", realtuntuliang_jin_1)

#################################2静态随机仿真###############################################
while Commurounds_jin_2 < rounds:
    print("--------------", "2静态随机仿真rounds", Commurounds_jin_2,"--------------")
    Commurounds_jin_2 += 1
    left_Nodes_jin_2 = all_Nodes_jin_2.copy()

    # 成功调度
    choose_node_jin_2 = weighted_random_sampling(all_Nodes_jin_2, miyou_jin_2)
    #left_Nodes.remove(choose_node)
    # 被调度的
    ran1 = random.random()
    if ran1 <= possi[choose_node_jin_2]:  # 通信成功
        for i in range(num):
            ran = random.random()
            #print(ran)
            if ran <= 10 * matrix[choose_node_jin_2][i]:# 共享成功
            #print("yes")
                AOI_jin_2[i] = 1
                realtuntuliang_jin_2[i] += 1
                left_Nodes_jin_2.remove(i)

    #print("left1",left_Nodes)
    for i in range(len(left_Nodes_jin_2)):
        AOI_jin_2[left_Nodes_jin_2[i]] += 1

    taltolAOI_jin_2 = [taltolAOI_jin_2[i] + AOI_jin_2[i] for i in range(len(AOI_jin_2))]

#################################3静态随机仿真###############################################
while Commurounds_jin_3 < rounds:
    print("--------------", "2静态随机仿真rounds", Commurounds_jin_3,"--------------")
    Commurounds_jin_3+= 1
    left_Nodes_jin_3 = all_Nodes_jin_3.copy()

    # 成功调度
    choose_node_jin_3 = weighted_random_sampling(all_Nodes_jin_3, miyou_jin_3)
    #left_Nodes.remove(choose_node)
    # 被调度的
    ran1 = random.random()
    if ran1 <= possi[choose_node_jin_3]:  # 通信成功
        for i in range(num):
            ran = random.random()
            #print(ran)
            if ran <= 10 * matrix[choose_node_jin_3][i]:# 共享成功
            #print("yes")
                AOI_jin_3[i] = 1
                realtuntuliang_jin_3[i] += 1
                left_Nodes_jin_3.remove(i)

    #print("left1",left_Nodes)
    for i in range(len(left_Nodes_jin_3)):
        AOI_jin_3[left_Nodes_jin_3[i]] += 1

    taltolAOI_jin_3 = [taltolAOI_jin_3[i] + AOI_jin_3[i] for i in range(len(AOI_jin_3))]
#################################1最大权重仿真###############################################
while Commurounds_max_1 < rounds:
    print("--------------", "1最大权重仿真rounds", Commurounds_max_1,"--------------")
    Commurounds_max_1 += 1
    left_Nodes_max_1 = []
    choose_node_max_1 = 0 # 被选择的
    # 成功调度
    for i in range(num):
        miyou_max_1[i]=(possi[i]/2)*AOI_max_1[i]*(AOI_max_1[i]+2)+possi[i]*max(Commurounds_max_1*Throughput[i]-realtuntuliang_max_1[i],0)
    for i in range(num):
        if miyou_max_1[i]== max(miyou_max_1):
            choose_node_max_1 = i
    ran = random.random()
    #print(choose_node_max_1)
    # 被调度的
    if ran <= possi[choose_node_max_1]:# 通信成功
            #print("yes")
            AOI_max_1[choose_node_max_1] = 1
            realtuntuliang_max_1[choose_node_max_1]+=1
    else: #失败
            #print("no")
            AOI_max_1[choose_node_max_1] += 1
    # 没被调度的
    left_Nodes_max_1 = all_Nodes_max_1.copy()
    left_Nodes_max_1.remove(choose_node_max_1)
    #print("left1",left_Nodes)
    for i in range(len(left_Nodes_max_1)):
        AOI_max_1[left_Nodes_max_1[i]] += 1

    taltolAOI_max_1 = [taltolAOI_max_1[i] + AOI_max_1[i] for i in range(len(AOI_max_1))]
    #print("aoi_max_1", AOI_max_1)
    #print("taltolaoi_max_1", taltolAOI_max_1)
    #print("吞吐量_max_1", realtuntuliang_max_1)

#################################2最大权重仿真###############################################
while Commurounds_max_2 < rounds:
    print("--------------", "2最大权重仿真rounds", Commurounds_max_2,"--------------")
    Commurounds_max_2 += 1
    left_Nodes_max_2 = all_Nodes_max_2.copy()
    choose_node_max_2=0
    # 成功调度
    for i in range(num):
        miyou_max_2[i]=0
        for j in range(num):
            miyou_max_2[i] += 1/2* matrix[i][j]*possi[j] * AOI_max_2[j] * (AOI_max_2[j] + 2)
    for i in range(num):
        if miyou_max_2[i] == max(miyou_max_2):
            choose_node_max_2 = i
    #left_Nodes.remove(choose_node)
    # 被调度的
    ran1 = random.random()
    if ran1 <= possi[choose_node_max_2]:  # 通信成功
        for i in range(num):
            ran = random.random()
            #print(ran)
            if ran <= 10 * matrix[choose_node_max_2][i]:# 共享成功
            #print("yes")
                AOI_max_2[i] = 1
                realtuntuliang_max_2[i] += 1
                left_Nodes_max_2.remove(i)

    #print("left1",left_Nodes)
    for i in range(len(left_Nodes_max_2)):
        AOI_max_2[left_Nodes_max_2[i]] += 1

    taltolAOI_max_2 = [taltolAOI_max_2[i] + AOI_max_2[i] for i in range(len(AOI_max_2))]
#################################3最大权重仿真###############################################
while Commurounds_max_3 < rounds:
    print("--------------", "3最大权重仿真rounds", Commurounds_max_3,"--------------")
    Commurounds_max_3 += 1
    left_Nodes_max_3 = all_Nodes_max_3.copy()
    choose_node_max_3=0
    # 成功调度
    for i in range(num):
        miyou_max_3[i]=0
        for j in range(num):
            miyou_max_3[i] += 1/2 * matrix[i][j] * possi[j] * AOI_max_3[j] * (AOI_max_3[j] + 2)\
                              + matrix[i][j]*possi[j]*max(Commurounds_max_3*Throughput[i]-realtuntuliang_max_3[i],0)
    for i in range(num):
        if miyou_max_3[i] == max(miyou_max_3):
            choose_node_max_3 = i
    #left_Nodes.remove(choose_node)
    # 被调度的
    ran1 = random.random()
    if ran1 <= possi[choose_node_max_3]:  # 通信成功
        for i in range(num):
            ran = random.random()
            #print(ran)
            if ran <= 10 * matrix[choose_node_max_3][i]:# 共享成功
            #print("yes")
                AOI_max_3[i] = 1
                realtuntuliang_max_3[i] += 1
                left_Nodes_max_3.remove(i)

    #print("left1",left_Nodes)
    for i in range(len(left_Nodes_max_3)):
        AOI_max_3[left_Nodes_max_3[i]] += 1

    taltolAOI_max_3 = [taltolAOI_max_3[i] + AOI_max_3[i] for i in range(len(AOI_max_3))]


print("-------------------------------------------------")
print("final jin 1",miyou_jin_1, "sum", S_jin_1)
print("final jin 2",miyou_jin_2, "sum",sum(miyou_jin_2),lambda1_jin_2)
print("final jin 3",miyou_jin_3, "sum",sum(miyou_jin_3),lambda1_jin_3)

avgAOI_jin_1 = [taltolAOI_jin_1[i]/Commurounds_jin_1 for i in range(len(taltolAOI_jin_1))]
#print(S_jin_1)
avgSysAOI_jin_1 = sum(avgAOI_jin_1)/num
print("avgaoi_jin_1", avgAOI_jin_1, 'avgSysAOI_jin_1',avgSysAOI_jin_1)

avgAOI_jin_2 = [taltolAOI_jin_2[i]/Commurounds_jin_2 for i in range(len(taltolAOI_jin_2))]
#print(S_jin_1)
avgSysAOI_jin_2 = sum(avgAOI_jin_2)/num
print("avgaoi_jin_2", avgAOI_jin_2, 'avgSysAOI_jin_2',avgSysAOI_jin_2)

avgAOI_jin_3 = [taltolAOI_jin_3[i]/Commurounds_jin_3 for i in range(len(taltolAOI_jin_3))]
#print(S_jin_1)
avgSysAOI_jin_3 = sum(avgAOI_jin_3)/num
print("avgaoi_jin_3", avgAOI_jin_3, 'avgSysAOI_jin_3',avgSysAOI_jin_3)

avgAOI_max_1 = [taltolAOI_max_1[i]/Commurounds_max_1 for i in range(len(taltolAOI_max_1))]
#print(S_max_1)
avgSysAOI_max_1 = sum(avgAOI_max_1)/num
print("avgaoi_max_1", avgAOI_max_1,'avgSysAOI_max_1',avgSysAOI_max_1)

avgAOI_max_2 = [taltolAOI_max_2[i]/Commurounds_max_2 for i in range(len(taltolAOI_max_2))]
#print(S_max_1)
avgSysAOI_max_2 = sum(avgAOI_max_2)/num
print("avgaoi_max_2", avgAOI_max_2,'avgSysAOI_max_2',avgSysAOI_max_2)

avgAOI_max_3 = [taltolAOI_max_3[i]/Commurounds_max_3 for i in range(len(taltolAOI_max_3))]
#print(S_max_1)
avgSysAOI_max_3 = sum(avgAOI_max_3)/num
print("avgaoi_max_3", avgAOI_max_3,'avgSysAOI_max_3',avgSysAOI_max_3)


#################################1静态随机理论值计算###############################################
aoi_jin_1 = 0
for i in range(len(possi)):
    c = alpha/(possi[i]*miyou_jin_1[i])
    #print(c)
    aoi_jin_1 += c
aoi_jin_1 = aoi_jin_1/num
print("expected_jin_1", aoi_jin_1)
#print("吞吐量", realtuntuliang)

#################################2静态随机理论值计算###############################################
matrix_jin_2 = np.array(list(matrix))
matrix_jin_2 *=10
for i in range(num):
    matrix_jin_2[i][i]=1
#print(matrix)
e_aoi_jin_2 = 0
for i in range(num):
    c2 = 0
    for j in range(num):
        c2 += miyou_jin_2[j]*matrix_jin_2[j][i]*possi[j]
    #print(1/c2)
    e_aoi_jin_2 += 1/c2
e_aoi_jin_2 = e_aoi_jin_2/num
print("expected_jin_2", e_aoi_jin_2)
#################################3静态随机理论值计算###############################################
matrix_jin_3 = np.array(list(matrix))
matrix_jin_3 *=10
for i in range(num):
    matrix_jin_3[i][i]=1
#print(matrix)
e_aoi_jin_3 = 0
for i in range(num):
    c3 = 0
    for j in range(num):
        c3 += miyou_jin_3[j]*matrix_jin_3[j][i]*possi[j]
    #print(1/c2)
    e_aoi_jin_3 += 1/c3
e_aoi_jin_3 = e_aoi_jin_3/num
print("expected_jin_3", e_aoi_jin_3)


debt_jin_1 = [max(Commurounds_jin_1*Throughput[i]-realtuntuliang_jin_1[i],0) for i in range(num)]
maxdebt_jin_1 = max(debt_jin_1)
print("吞吐debt_jin_1", debt_jin_1,"max吞吐debt_jin_1", maxdebt_jin_1, realtuntuliang_jin_1)

debt_jin_2 = [max(Commurounds_jin_2*Throughput[i]-realtuntuliang_jin_2[i],0) for i in range(num)]
maxdebt_jin_2 = max(debt_jin_2)
print("吞吐debt_jin_2", debt_jin_2,"max吞吐debt_jin_2", maxdebt_jin_2, realtuntuliang_jin_2)

debt_jin_3 = [max(Commurounds_jin_3*Throughput[i]-realtuntuliang_jin_3[i],0) for i in range(num)]
maxdebt_jin_3 = max(debt_jin_3)
print("吞吐debt_jin_3", debt_jin_3,"max吞吐debt_jin_3", maxdebt_jin_3, realtuntuliang_jin_3)

debt_max_1 = [max(Commurounds_max_1*Throughput[i]-realtuntuliang_max_1[i],0) for i in range(num)]
maxdebt_max_1 = max(debt_max_1)
print("吞吐debt_max_1", debt_max_1, "max吞吐debt_max_1", maxdebt_max_1,realtuntuliang_max_1)

debt_max_2 = [max(Commurounds_max_2*Throughput[i]-realtuntuliang_max_2[i],0) for i in range(num)]
maxdebt_max_2 = max(debt_max_2)
print("吞吐debt_max_2", debt_max_2, "max吞吐debt_max_2", maxdebt_max_2,realtuntuliang_max_2)

debt_max_3 = [max(Commurounds_max_3*Throughput[i]-realtuntuliang_max_3[i],0) for i in range(num)]
maxdebt_max_3 = max(debt_max_3)
print("吞吐debt_max_3", debt_max_3, "max吞吐debt_max_3", maxdebt_max_3,realtuntuliang_max_3)

file_path = 'delta0.9_num20.txt'
with open(file_path, mode='w', encoding='utf-8') as file_obj:
    file_obj.writelines(["num=", str(num)])
    file_obj.writelines(["\ndelta=",str(delta) , " possi", str(possi)])
    file_obj.writelines(["\n", str(matrix)])
    file_obj.writelines("\n 随机静态最优策略")
    file_obj.writelines(["\nfinal jin 1 sum=", str(S_jin_1), " final jin 1=", str(miyou_jin_1)])
    file_obj.writelines(["\nfinal jin 2 sum=", str(sum(miyou_jin_2)), " final jin 2=", str(miyou_jin_2),str(lambda1_jin_2)])
    file_obj.writelines(["\nfinal jin 3 sum=", str(sum(miyou_jin_3)), " final jin 3=", str(miyou_jin_3), str(lambda1_jin_3)])
    file_obj.writelines("\n 随机静态最优策略AOI")
    file_obj.writelines(["\navgSysAOI_jin_1=", str(avgSysAOI_jin_1), " avgaoi_jin_1=", str(avgAOI_jin_1)])
    file_obj.writelines(["\navgSysAOI_jin_2=", str(avgSysAOI_jin_2), " avgaoi_jin_2=", str(avgAOI_jin_2)])
    file_obj.writelines(["\navgSysAOI_jin_3=", str(avgSysAOI_jin_3), " avgaoi_jin_3=", str(avgAOI_jin_3)])
    file_obj.writelines(["\nexpected_jin_1=", str(aoi_jin_1)])
    file_obj.writelines(["\nexpected_jin_2=", str(e_aoi_jin_2)])
    file_obj.writelines(["\nexpected_jin_3=", str(e_aoi_jin_3)])
    file_obj.writelines("\n 最大权重策略AOI")
    file_obj.writelines(["\navgSysAOI_max_1=", str(avgSysAOI_max_1), " avgaoi_max_1=", str(avgAOI_max_1)])
    file_obj.writelines(["\navgSysAOI_max_2=", str(avgSysAOI_max_2), " avgaoi_max_2=", str(avgAOI_max_2)])
    file_obj.writelines(["\navgSysAOI_max_3=", str(avgSysAOI_max_3), " avgaoi_max_3=", str(avgAOI_max_3)])
    file_obj.writelines("\n 吞吐量")
    file_obj.writelines(["\nmax吞吐debt_jin_1=", str(maxdebt_jin_1), " 吞吐debt_jin_1=", str(debt_jin_1), str(realtuntuliang_jin_1)])
    file_obj.writelines(["\nmax吞吐debt_jin_2=", str(maxdebt_jin_2), " 吞吐debt_jin_2=", str(debt_jin_2), str(realtuntuliang_jin_2)])
    file_obj.writelines(["\nmax吞吐debt_jin_3=", str(maxdebt_jin_3), " 吞吐debt_jin_3=", str(debt_jin_3), str(realtuntuliang_jin_3)])
    file_obj.writelines(["\nmax吞吐debt_max_1=", str(maxdebt_max_1), " 吞吐debt_max_1=", str(debt_max_1), str(realtuntuliang_max_1)])
    file_obj.writelines(["\nmax吞吐debt_max_2=", str(maxdebt_max_2), " 吞吐debt_max_2=", str(debt_max_2), str(realtuntuliang_max_2)])
    file_obj.writelines(["\nmax吞吐debt_max_3=", str(maxdebt_max_3), " 吞吐debt_max_3=", str(debt_max_3), str(realtuntuliang_max_3)])

