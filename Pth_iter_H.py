import os
import sys

gen = os.path.dirname(os.path.abspath(__file__))
dir1 = 'som-tsp'
dir2 = 'src'

ref_dir = os.path.join(gen, dir1, dir2)
sys.path.append(ref_dir)
import main

import numpy as np
import matplotlib.pyplot as plt
import random
import math
from datetime import datetime
from pyswarm import pso
from sko.PSO import PSO
import pickle
from collections import deque   # 双向队列
import copy
from scipy.linalg import solve
import time
import csv
import scipy.optimize as sco

''' 
SOM 部分 
'''
def tsp_generate():
    ''' 想法：将整个区域划分为 (根号N * beta1) * (根号N * beta2) 个网格，1 < beta1, beta2 < 2，
        beta1,2 随机，结果向上取整，
        每个网格内随机分布消息节点、窃听节点、辅助节点，
        一个网格内最多3个节点 '''

    global S_X, S_Y, R_X, R_Y, E_X, E_Y, _tsp_path, Fading
    " 网格数 "
    beta1 = random.random() + 1
    beta2 = random.random() + 1
    x_num = math.ceil(math.ceil(math.sqrt(N)) * beta1)
    y_num = math.ceil(math.ceil(math.sqrt(N)) * beta2)
    " 每个网格的大小 "
    x_size = scope / x_num
    y_size = scope / y_num
    " 选中的网格 "
    choose = random.sample(range(0, x_num * y_num), N)
    " choose 转化成二维网格坐标 "
    choose_coor = []
    
    for i in range(0, N):
        coor = divmod(choose[i], x_num)     # 计算网格位置 第 coor[0] 行 coor[1] 列， 0为起始
        choose_coor.append(coor)
        " 生成每个网格内节点坐标 "
        x0 = x_size * coor[1]    # 该网格左下角坐标
        y0 = y_size * coor[0]

        x = np.random.random(3) * x_size + x0
        y = np.random.random(3) * y_size + y0

        " 调整"
        x, y = adjust(x, y)    

        S_X.append(x[0])
        S_Y.append(y[0])
        R_X.append(x[1])
        R_Y.append(y[1])
        E_X.append(x[2])
        E_Y.append(y[2])


        " 生成每个网格内地面节点之间的fading，服从Rayleigh分布 顺序为 S R E"
        fading = np.random.rayleigh(scale = 0.5, size=(3, 3))
        fading = np.triu(fading, 1)  # 保留上三角部分，同时去除对角线元素
        fading += fading.T  # 将上三角"拷贝"到下三角部分

        " 判断"
        while True:
            detect = np.array(fading)
            bool =  len(np.where(detect > 1)[0])
            if bool == 0: 
                break
            fading = np.random.rayleigh(scale = 0.5, size=(3, 3))
            fading = np.triu(fading, 1) 
            fading += fading.T


        Fading.append(fading.tolist())

    _tsp_path = './assets/Pth_iter_H'     

    if not os.path.exists(_tsp_path):
        os.makedirs(_tsp_path)

    with open(os.path.join(_tsp_path, "test.tsp"), 'w', encoding='utf-8') as f:
        f.write('NAME : test100\nTYPE : TSP\n')
        f.write('DIMENSION : {}'.format(N))
        f.write('\nEDGE_WEIGHT_TYPE : EUC_2D\nNODE_COORD_SECTION\n')
        for i, _ in enumerate(S_X):
            f.write('{} {} {}\n'.format(i+1, S_X[i], S_Y[i]))
        f.close()


def adjust(x, y):
    x_copy = copy.copy(x)
    y_copy = copy.copy(y)
    a = np.array([x[0], y[0]])
    b = np.array([x[1], y[1]])
    c = np.array([x[2], y[2]])
    ab = distance(a, b)
    bc = distance(b, c)
    ac = distance(a, c)
    _min = min(ab, bc, ac)

    if ac == _min:
        x[2] = x_copy[1]
        y[2] = y_copy[1]
        x[1] = x_copy[2]
        y[1] = y_copy[2]
    elif bc == _min:
        x[2] = x_copy[0]
        y[2] = y_copy[0]
        x[0] = x_copy[2]
        y[0] = y_copy[2]

    return x, y

'''
画散点图
'''
def plot(x_num, y_num):    
    plt.scatter(S_X, S_Y, s = 5, c = 'blue')
    plt.scatter(R_X, R_Y, s = 5, c = 'green')
    plt.scatter(E_X, E_Y, s = 5, c = 'red')    

    grid_x = np.linspace(0,scope, x_num + 1)
    grid_y = np.linspace(0,scope, y_num + 1)

    for i in grid_x:
        plt.axvline(x=i,ls="-",c="gray", linewidth=0.3)#添加垂直直线
    for j in grid_y:
        plt.axhline(y=j,ls="-",c="gray", linewidth=0.3)#添加水平直线

    plt.show()


" 调试用 "
def plot_tri(s, r, e, u):
    plt.scatter(s[0], s[1], s = 10, c = 'blue')
    plt.scatter(r[0], r[1], s = 10, c = 'green')
    plt.scatter(e[0], e[1], s = 10, c = 'red')   
    plt.scatter(u[0], u[1], s = 10, c = 'black') 

    plt.plot([s[0],r[0]],[s[1],r[1]])
    plt.plot([s[0],e[0]],[s[1],e[1]])
    plt.plot([r[0],e[0]],[r[1],e[1]])


''' 
PSO 部分
'''

"节点距离公式"
def distance(x, y):
    d = np.linalg.norm(x-y)
    return d

#  信噪比转换，从dB转为比值
def shannon(snr):
    return 10**(snr/10)


" 信道衰减因子计算 "
def h(x, y, f):
    d = distance(x, y)
    if x[2] > 0 or y[2] > 0:   # 如果是空地节点之间
        return (shannon(zetaA)*((d/d0)**(-kappa)))
    else:       # 如果是地面节点之间 
        return (shannon(zetaB)*((d/d0)**(-kappa))* f)    # 乘以一个fading，服从CN(0,1)


" 目标函数计算公式 "
def energy(variable):
    # temp = np.array([variable[0], variable[1], H])      # 无人机悬停点
    temp = variable
    # 飞行能耗
    t = distance(temp, lsp) / V     # 从上一个位置飞到悬停位置所需时间
    E_fly = P_V * t
    # 悬停能耗
    E_hover = P_h * T
    # 通信能耗
    E_comm = P_receive * (1 - rho) * T
    # WPT能耗
    E_wpt = P_U * rho * T

    E_total = E_fly + E_hover + E_comm + E_wpt

    return E_total

def energy2(variable):
    temp = variable

    R_SEC = RSEC_alpha(variable)

    # 飞行能耗
    t = distance(temp, lsp) / V     # 从上一个位置飞到悬停位置所需时间
    E_fly = P_V * t
    # 悬停能耗
    E_hover = P_h * T
    # 通信能耗
    E_comm = P_receive * (1 - rho) * T
    # WPT能耗
    E_wpt = P_U * rho * T

    E_total = E_fly + E_hover + E_comm + E_wpt

    " 加惩罚 "
    if R_SEC < R_SEC_bound:
        fine = (R_SEC_bound - R_SEC) / R_SEC_bound
        # fine = -(R_SEC_bound - R_SEC) / R_SEC
    else:
        fine = 0

    A = E_total * (1+fine)

    return A

    # if R_SEC < 0:
    #     E_total = np.inf
    # return E_total


" 安全容量计算公式  协作通信时隙  最优alpha  P_R 定，确定alpha之后，再调整UAV WPT的功率 "
def RSEC_alpha(variable): 
    u = variable
    V_R, V_U, V_E = PU_V_obtain(u)
    R_SEC = (1-rho) * V_R * (V_U - V_E) / (V_R + V_U)
    return R_SEC


def PU_V_obtain(u):
    global P_U
    V_R, V_U, V_E = V_obtain(P_R, u)
    alpha = V_R / (V_R + V_U)
    W_need = (1 - alpha) * (1 - rho) * T * P_R      # 辅助节点发送消息所需能量
    if W_need > P_th:   
        W_need = P_th
        P_R_revise = W_need / (1 - alpha) / (1 - rho) / T   # 若超过P_th，则重新计算各节点的传输速率 V
        V_R, V_U, V_E = V_obtain(P_R_revise, u)
    P_U = W_need / eta  # 计算UAV WPT所需功率
    return V_R, V_U, V_E


def V_obtain(P_R, u):
    V_R = W * np.log2(1 + P * (h(s, r, fading[0][1])**2)/sigma/W)
    V_U = W * np.log2(1 + P * (h(s, u, 0)**2)/sigma/W + P_R * (h(r, u, 0)**2)/sigma/W)
    V_E = W * np.log2(1 + P * (h(s, e, fading[0][2])**2)/sigma/W + P_R * (h(r, e, fading[1][2])**2)/sigma/W)
    return V_R, V_U, V_E


" 求R的发送功率，与rho相关 "
def PR_obtain(r, u):
    E_H = eta * P_U   # 一个时隙T预期能收集的能量
    if E_H > P_th:
        E_H = P_th
    P_R = 2 * E_H * rho / (1 - rho)
    return P_R
    

" 确定PSO变量的上下界 "
def bound_decision(s, r, e):

    x_lb = min(s[0],r[0],e[0])
    x_ub = max(s[0],r[0],e[0])
    y_lb = min(s[1],r[1],e[1])
    y_ub = max(s[1],r[1],e[1])


    # 变量为 x, y   若变量不止x,y 可以继续加
    lb = [x_lb, y_lb, H_lb]
    ub = [x_ub, y_ub, H_ub]
    return lb, ub


" 找外接圆的圆心和半径  外接圆到三角形的三个点距离相等 "
def get_outer_circle(A, B, C):
    xa, ya = A[0], A[1]
    xb, yb = B[0], B[1]
    xc, yc = C[0], C[1]

    # 两条边的中点
    x1, y1 = (xa + xb) / 2.0, (ya + yb) / 2.0
    x2, y2 = (xb + xc) / 2.0, (yb + yc) / 2.0

    # 两条线的斜率
    ka = (yb - ya) / (xb - xa) if xb != xa else None
    kb = (yc - yb) / (xc - xb) if xc != xb else None

    alpha = np.arctan(ka) if ka != None else np.pi / 2
    beta = np.arctan(kb) if kb != None else np.pi / 2

    # 两条垂直平分线的斜率
    k1 = np.tan(alpha + np.pi / 2)
    k2 = np.tan(beta + np.pi / 2)

    # 圆心
    y, x = solve([[1.0, -k1], [1.0, -k2]], [y1 - k1 * x1, y2 - k2 * x2])
    # 半径
    r1 = np.sqrt((x - xa)**2 + (y - ya)**2)

    return (x, y, r1)


" 求两个圆的交点 "
def insec(p1, r1, p2, r2):
    x = p1[0]
    y = p1[1]
    R = r1
    a = p2[0]
    b = p2[1]
    S = r2
    d = math.sqrt((abs(a - x)) ** 2 + (abs(b - y)) ** 2)
    if d > (R + S) or d < (abs(R - S)):
        #print("Two circles have no intersection")
        return None, None
    elif d == 0:
        #print("Two circles have same center!")
        return None, None
    else:
        A = (R ** 2 - S ** 2 + d ** 2) / (2 * d)
        h = math.sqrt(R ** 2 - A ** 2)
        x2 = x + A * (a - x) / d
        y2 = y + A * (b - y) / d

        x3 = round(x2 - h * (b - y) / d, 2)
        y3 = round(y2 + h * (a - x) / d, 2)
        x4 = round(x2 + h * (b - y) / d, 2)
        y4 = round(y2 - h * (a - x) / d, 2)
        c1 = [x3, y3]
        c2 = [x4, y4]
        return c1, c2


def pso_func(s, r, e):
    lower_bound, upper_bound = bound_decision(s, r, e)

    constraint_ueq2 = (
        lambda variable: R_SEC_bound-RSEC_alpha(variable)
        ,
    )
    pso = PSO(func=energy2, constraint_ueq=constraint_ueq2, n_dim=variable_num, pop=population, max_iter=max_iteration, lb=lower_bound, ub=upper_bound, w=w_p, c1=c1_p, c2=c2_p,verbose=False)
    pso.run(precision=precision_p)
    print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)

    # 返回所求的变量 及 值
    variable = [pso.gbest_x, pso.gbest_y]
    return variable


def plan_B():
    r11 = shannon(zetaA)/shannon(zetaB) * (distance(s, e)**2) - H_lb ** 2   # 消息节点S的取值半径，在此半径内，h_su > h_se
    r1 = math.sqrt(r11)
    r22 = shannon(zetaA)/shannon(zetaB) * (distance(r, e)**2) - H_lb ** 2   # 辅助节点R的取值半径，在此半径内，h_ru > h_re
    r2 = math.sqrt(r22)

    c1, c2 = insec(s, r1, r, r2)    # 获得两个圆的相交点
    if c1 == None:      # 如果没有交点，则取s和r的中点
        x = (s[0] + r[0]) / 2
        y = (s[1] + r[1]) / 2
        var = np.array([x, y, H_lb])
    else:
        c11 = np.array([c1[0], c1[1], H_lb])
        c22 = np.array([c2[0], c2[1], H_lb])
        d1 = distance(lsp, c11)
        d2 = distance(lsp, c22)
        if d1 < d2:
            var = c11
        elif d1 > d2:
            var = c22
        else:
            var = (c11 + c22) / 2
    return var


def plan_C():
    lower_bound, upper_bound = bound_decision(s, r, e)
    pso = PSO(func=energy2, n_dim=variable_num, pop=population, max_iter=max_iteration, lb=lower_bound, ub=upper_bound, w=w_p, c1=c1_p, c2=c2_p,verbose=False)
    pso.run(precision=precision_p)
    print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
    variable = [pso.gbest_x, pso.gbest_y]
    return variable


def proposed():
    global route, lsp, H_lb, E, R_SEC
    print('\n-----------------------------------本文算法-----------------------------------')
    var, val = pso_func(s, r, e)

    if val == np.inf:
        temp = H_lb     
        H_lb = 1      
        var, val = pso_func(s, r, e)       
        H_lb = temp     


        if val == np.inf:
            var, val = plan_C()

    E_coms = energy(var)    # UAV在当前消息节点消耗的能量
    print('能量',E_coms)
    lsp = var
    route.append(lsp)       # UAV的行进路线

    Rsec = RSEC_alpha(var)
    print('安全容量', Rsec)

    E.append(E_coms)
    R_SEC.append(Rsec)

    get_Wneed(var)


def get_Wneed(u):
    V_R, V_U, V_E = V_obtain(P_R, u)
    alpha = V_R / (V_R + V_U)
    W_need = (1 - alpha) * (1 - rho) * T * P_R
    print("P_th: ", W_need)


" 对比算法需要 "
def energy_comp(variable, args):

    temp = np.array([args[0], args[1], variable.item()])
    # 飞行能耗
    t = distance(temp, args[2]) / V     # 从上一个位置飞到悬停位置所需时间
    E_fly = P_V * t
    # 悬停能耗
    E_hover = P_h * T
    # 通信能耗
    E_comm = P_receive * (1 - rho) * T
    # WPT能耗
    E_wpt = P_U * rho * T

    E_total = E_fly + E_hover + E_comm + E_wpt

    return E_total


" 对比算法需要 "
def RSEC_alpha_comp(variable, args): 
    u = np.array([args[0], args[1], variable.item()])
    V_R, V_U, V_E = PU_V_obtain(u)
    R_SEC = (1-rho) * V_R * (V_U - V_E) / (V_R + V_U)
    return R_SEC


" sco point 为 S, R, SR中心  默认采用 Sequential Least Squares Programming 方法"
def slsqp(point, _lsp, _route, _E, _R_SEC):
    arg = [point[0], point[1], _lsp]     # 辅助变量  计划悬停的位置以及上一悬停点
    cons = ({'type':'ineq', 'fun': lambda x: RSEC_alpha_comp(x, arg) - R_SEC_bound},
            {'type':'ineq', 'fun': lambda x: x - H_lb},
            {'type':'ineq', 'fun': lambda x: H_ub - x})
    opt = sco.minimize(fun=energy_comp,x0=H_lb,args=(arg), constraints=cons)

    print(opt)

    E_coms = opt.fun    # UAV在当前消息节点消耗的能量
    print('能量',E_coms)

    var = np.array([point[0], point[1], opt.x.item()])      # sco求出来的悬停点
    Rsec = RSEC_alpha_comp(opt.x, point)
    print('安全容量', Rsec)

    _lsp = var
    _route.append(_lsp)

    _E.append(E_coms)
    _R_SEC.append(Rsec)

    return _lsp

" 对比算法1 无人机悬停在消息节点 S 上方 "
def comp1():
    global lsp1
    print('\n-----------------------------------对比算法1-----------------------------------')
    lsp1 = slsqp(s, lsp1, route1, E_1, R_SEC1)      # 使用scipy优化


" 对比算法2 无人机悬停在辅助节点 R 上方 "
def comp2():
    global lsp2
    print('\n-----------------------------------对比算法2-----------------------------------')
    lsp2 = slsqp(r, lsp2, route2, E_2, R_SEC2)


" 对比算法3 无人机悬停在 SR 中心 "
def comp3():
    global lsp3
    point = (s + r) / 2     # SR中心
    print('\n-----------------------------------对比算法3-----------------------------------')
    lsp3 = slsqp(point, lsp3, route3, E_3, R_SEC3)


def value_copy(e_sum, e_all, rsec_sum, rsec_all, e, r_sec):
    temp = []
    e_sum.append(temp)
    e_sum[-1] = copy.deepcopy(e)
    e_all.append(temp)
    e_all[-1] = copy.deepcopy(e)
    rsec_sum.append(temp)
    rsec_sum[-1] = copy.deepcopy(r_sec)
    rsec_all.append(temp)
    rsec_all[-1] = copy.deepcopy(r_sec)


# 存放每次重复实验（行）  每个训练节点的总能耗（列）
def save():
    
    value_copy(E_sum, E_all, RSEC_sum, RSEC_all, E, R_SEC)
    value_copy(E_sum1, E_all1, RSEC_sum1, RSEC_all1, E_1, R_SEC1)
    value_copy(E_sum2, E_all2, RSEC_sum2, RSEC_all2, E_2, R_SEC2)
    value_copy(E_sum3, E_all3, RSEC_sum3, RSEC_all3, E_3, R_SEC3)


# 求每次重复实验的平均值，最大值，最小值等
def ave(e_sum):
    SUM = []
    for ee in e_sum:
        SUM.append(sum(ee))
    '''可以用于工字图制作'''
    MIN = min(SUM)  # 最小值
    MAX = max(SUM)  # 最大值
    AVE = np.mean(SUM)  # 平均值

    result = [MIN, MAX, AVE]
    return result


# 求安全容量 平均合格率
def percent(rsec_sum):
    result = ave(rsec_sum)
    Q = []
    for rsec in rsec_sum:
        num = 0
        for sec in rsec:
            if sec >= R_SEC_bound:
                num += 1
        Q.append(num / N)
    average = np.mean(Q)
    final = result + [average]
    return final


# 将要用于绘图的数据存储到RESULT，以便存入文件
def output():
    global E_result, RSEC_result

    E_result.append(ave(E_sum))
    E_result.append(ave(E_sum1))
    E_result.append(ave(E_sum2))
    E_result.append(ave(E_sum3))
    E_result.append([])

    RSEC_result.append(percent(RSEC_sum))
    RSEC_result.append(percent(RSEC_sum1))
    RSEC_result.append(percent(RSEC_sum2))
    RSEC_result.append(percent(RSEC_sum3))
    RSEC_result.append([])


# 每次重复实验前，将矩阵重置
def reset():
    global S_X, S_Y, R_X, R_Y, E_X, E_Y
    global E, R_SEC, route
    global E_1, R_SEC1, route1
    global E_2, R_SEC2, route2
    global E_3, R_SEC3, route3
    S_X = []
    S_Y = []
    R_X = []
    R_Y = []
    E_X = []
    E_Y = []

    route = []
    E = []
    R_SEC = []

    route1 = []
    E_1 = []
    R_SEC1 = []

    route2 = []
    E_2 = []
    R_SEC2 = []

    route3 = []
    E_3 = []
    R_SEC3 = []


# 每次迭代之前，部分变量需要清除数据
def _clear():
    global E_sum, E_sum1, E_sum2, E_sum3
    global RSEC_sum, RSEC_sum1, RSEC_sum2, RSEC_sum3
    global E_all, E_all1, E_all2, E_all3
    global RSEC_all, RSEC_all1, RSEC_all2, RSEC_all3
    E_sum.clear()
    E_sum1.clear()
    E_sum2.clear()
    E_sum3.clear()
    RSEC_sum.clear()
    RSEC_sum1.clear()
    RSEC_sum2.clear()
    RSEC_sum3.clear()
    E_all.append([])
    E_all1.append([])
    E_all2.append([])
    E_all3.append([])
    RSEC_all.append([])
    RSEC_all1.append([])
    RSEC_all2.append([])
    RSEC_all3.append([])


# 输出到文件
class Logger(object):
    def __init__(self, file_path: str = "./Default.log"):
        self.terminal = sys.stdout
        self.log = open(file_path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


if __name__ == '__main__':

    start = time.localtime()
    sys.stdout = Logger('./Pth_log.txt')

    ''' 创建tsp文件 '''
    Pth_iter = [0.4, 0.6, 0.8, 1, 1.2, 1.4]
    ex_num = 50       # 重复实验的次数
    N = 30     # 消息节点数量  总节点数量将是3倍，需包括辅助节点和窃听节点
    scope = 1000     # 区域范围
    W = 4e7     # 带宽  40M Hz
    
    d0 = 1
    kappa = 2  # path loss exponent
    zetaA = -40  # path loss constant（dB） 空地节点之间
    zetaB = -30  # path loss constant（dB） 地面节点之间
    sigma = 1e-13  # 高斯白噪声（-100dBm）≈1e-13   -110dBm ≈1e-16
    T = 1       # 时隙 1s
    Fading = []     # small scale fading 服从 CN(0, 1) 瑞利分布
    P_receive = 0.5       # 无人机接收消息功率 50 mW  改大一点为 0.5 W

    " 无人机飞行能耗公式需要 "
    delta = 0.012   # profile drag coefficient
    rho_UAV = 1.225     # air density in kg/m^3
    s_UAV = 0.05    # Rotor solidity, define as the ratio of the total blade area bcR to the disc area A, 
                    # s = bc/(pi*R)  b = 4 number of blades  c = 0.0157 blade or aerofoil chord length
    R = 0.4     # Rotor radius in meter
    A = 0.503   # Rotor disc area in m^2  A = pi * R^2
    omega = 300     # blade angular velocity in radians/second
    k = 0.1     # incremental correction factor to induced power
    W_UAV = 20      # aircraft weight in Newton
    U_tip = 120     # tip speed of the rotor blade (m/s) U_tip = omega * R
    v_0 = 4.03       # mean rotor induced velocity in hover
    d_0 = 0.6       # fuselage drag ratio, d_0 = S_FP / (s_UAV * A)
                    # S_FP = 0.0151  fuselage equivalent flat plate area in m^2
    

    " 可变 变量 "
    V = 10      # 无人机飞行速度 m/s
    R_SEC_bound = 0.01
    eta = 0.6   #! 能量转换效率，未确定
    # P_th = 1     #! 无人机WPT最大输出功率 未定 可变
    P = 2      # 消息节点发送功率
    P_R = 2    # 辅助节点发送功率  
    P_U = 5    # 无人机WPT功率   已改   根据辅助节点所需能量调整
    ' 无人机飞行高度上下界 '
    H = 10      # H 不为所求变量时的值
    H_lb = 5
    H_ub = 20
    ' 能量收集时隙上下界 '
    rho = 0.4   # rho 不为所求变量时的值
    rho_lb = 0.1
    rho_ub = 0.9
    ' PSO算法的变量 '
    population = 200    # 群体数量
    variable_num = 3    # 所求变量个数
    max_iteration = 50     # 迭代次数
    w_p = 0.9        # 前速度，惯性项
    c1_p = 0.8      # 认知部分加速度常数
    c2_p = 0.2     # 社会部分加速度常数
    precision_p = 1e-7   # 精度

    S_X = []    # 消息节点坐标
    S_Y = []
    R_X = []    # 辅助节点坐标
    R_Y = []
    E_X = []    # 窃听节点坐标
    E_Y = []

    '  结果输出 '
    E_result = []
    RSEC_result = []

    ' 提出的算法 '
    route = []  # 无人机飞行路径
    E = []      # 记录经过每个消息节点消耗的能量
    R_SEC = []  # 记录消息节点的安全容量
    E_sum = []  # 记录ex_num次重复实验的能耗值
    RSEC_sum = []   # 记录ex_num次重复实验的安全容量值
    E_all = []  # 记录 ex_num * iter 次数 所有能耗值，便于后续查看
    RSEC_all = []  # 记录 ex_num * iter 次数 所有安全容量值，便于后续查看

    ' 对比算法1 '
    route1 = []
    E_1 = []
    R_SEC1 = []
    E_sum1 = []
    RSEC_sum1 = []
    E_all1 = []
    RSEC_all1 = []

    ' 对比算法2 '
    route2 = []
    E_2 = []
    R_SEC2 = []
    E_sum2 = []
    RSEC_sum2 = []
    E_all2 = []
    RSEC_all2 = []

    ' 对比算法3 '
    route3 = []
    E_3 = []
    R_SEC3 = []
    E_sum3 = []
    RSEC_sum3 = []
    E_all3 = []
    RSEC_all3 = []

    _tsp_path = ''

    " 无人机飞行以及悬停功率 "
    P_0 = delta / 8 * rho_UAV * s_UAV * A * (omega ** 3) * (R ** 3)
    P_i = (1 + k) * W_UAV * np.sqrt(W_UAV) / np.sqrt(2 * rho_UAV * A)
    P_h = P_0 + P_i     # 无人机悬停功率
    P_V = P_0 * (1 + 3 * (V ** 2) / (U_tip ** 2)) + P_i * v_0 / V + 1/2 * d_0 * rho_UAV * s_UAV * A * (V ** 3)

    
    for P_th in Pth_iter:
        _clear()
        for ex in range(ex_num):
            reset()
            " 用 SOM 求出消息节点之间的最短路径 "
            tsp_generate()
            tsp_dir = os.path.join(_tsp_path, "test.tsp")  # tsp文件的存放位置
            seq = main.tsp(tsp_dir).values   # SOM 得到的消息节点访问顺序

            " 获取 S_X 与 route 之间的偏移量"
            initial = S_X.index(min(S_X))     # 最左边的消息节点索引
            offset = np.where(seq == initial)[0][0]     # np.where 返回的是 tuple

            lsp = np.array([S_X[initial], S_Y[initial], H])    # 无人机的上一个位置坐标，在这里表示出发点
            route.append(lsp)

            lsp1 = np.array([S_X[initial], S_Y[initial], H])
            route1.append(lsp1)

            lsp2 = np.array([S_X[initial], S_Y[initial], H])
            route2.append(lsp2)

            lsp3 = np.array([S_X[initial], S_Y[initial], H])
            route3.append(lsp3)


            " 每个消息节点采用PSO求最优的UAV悬停点 "
            for i in range(N):
                index = seq[(offset + i) % N]    # 消息节点, i 加 偏移量 取 N 模
                fading = Fading[index]
                " PSO问题中各个节点坐标 [x, y, H]"
                s = np.array([S_X[index], S_Y[index], 0])
                r = np.array([R_X[index], R_Y[index], 0])
                e = np.array([E_X[index], E_Y[index], 0])

                # # 提出的算法
                proposed()
                # 对比算法1
                comp1()
                # 对比算法2
                comp2()
                # 对比算法3
                comp3()
                # print('a')

            save()

        output()


    '''Pth_iter 存储用于绘图的数据'''
    f = open('Pth_iter_H_energy_50.csv', 'w', encoding='utf-8', newline='')
    writer = csv.writer(f)
    writer.writerows(E_result)
    f.close()

    f = open('Pth_iter_H_rsec_50.csv', 'w', encoding='utf-8', newline='')
    writer = csv.writer(f)
    writer.writerows(RSEC_result)
    f.close()

    " 查错用 "
    f = open('Pth_iter_e_all_50.csv', 'w', encoding='utf-8', newline='')
    writer = csv.writer(f)
    writer.writerows(E_all)
    f.close()

    f = open('Pth_iter_e_all1_50.csv', 'w', encoding='utf-8', newline='')
    writer = csv.writer(f)
    writer.writerows(E_all1)
    f.close()

    f = open('Pth_iter_e_all2_50.csv', 'w', encoding='utf-8', newline='')
    writer = csv.writer(f)
    writer.writerows(E_all2)
    f.close()

    f = open('Pth_iter_e_all3_50.csv', 'w', encoding='utf-8', newline='')
    writer = csv.writer(f)
    writer.writerows(E_all3)
    f.close()


    # 输出开始结束时间
    end = time.localtime()
    print('start:')
    print(time.strftime('%Y-%m-%d %H:%M:%S', start))
    print('end:')
    print(time.strftime('%Y-%m-%d %H:%M:%S', end))


