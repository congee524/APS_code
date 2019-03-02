import random


#  假设一个未编码的个体表示为：取,取,不取,不取，可使用10进制数12表示
def encode(N, unit):  #  N：染色体长度（如4）；unit：个体表示（如12）
    unit = int(unit)
    unit_str = str(bin(unit))[2:].zfill(N)  # 左侧补0
    unit_list = []
    for s in unit_str:
        unit_list.append(s)
    return unit_list
def decode(unit_list):
    l = ll = len(unit_list) - 1
    c = 0
    while l>=0:
        if unit_list[l] == '1':
            c +=  pow(2, ll - l)
        l -= 1
    return c

# 计算种群的适应性概率
def getRWSPList(population, w, v, W):  # population：总群；w：物体重量list；v：物体价值list；W：背包的重量阈值
    n = len(population)  # 群体总数
    v_list = []  # 价值list
    for i in population:
        unit_code = encode(N, i)  # 获得编码
        unit_w = 0  # 个体的总量
        unit_v = 0  # 个体的价值
        for j in range(N):
            unit_w += int(unit_code[j]) * w[j]
            unit_v += int(unit_code[j]) * v[j]
        if unit_w <= W:
            v_list.append(unit_v)
        else:
            v_list.append(0)  # 超重
    p_list = []  # 每一个个体的概率
    v_all = sum(v_list)
    for i in range(n):
        p_list.append(v_list[i] * 1.0 / v_all)
    return p_list

# 根据适应性概率随机选择一个个体
def RWS(population, plist):  # plist为总群个体抽中概率list
    random.seed()
    r = random.random()  # 获得随机数
    c = 0
    # print plist, r
    for (index, item) in enumerate(plist):
        c += item
        if r < c:
            return population[index]

#  获得随机couple组
def getRandomCouple(n):  # n:个体总数
    random.seed()
    selected = [0]*n  # 是否被选择了
    couples = []  # 配对list
    for i in range(n//2):
        pair = []
        while len(pair) < 2:
            unit_index = random.randint(0, n-1)
            if not selected[unit_index]:
                pair.append(unit_index)
                selected[unit_index] = True
        couples.append(pair)
    return couples

def crossover(population, couples, cross_p, N):  # cross_p为交叉概率;N为编码长度
    random.seed()
    new_population = []
    for pair in couples:
        unit_one = encode(N, population[pair[0]])
        unit_two = encode(N, population[pair[1]])
        p = random.random()
        if p >= (1 - cross_p):
            # 交叉使用从随机位置交叉尾部
            random_loc = random.randint(0, N-1)  # 获得随机位置
            new_population.append(unit_one[0:random_loc] + unit_two[random_loc:])
            new_population.append(unit_two[0:random_loc] + unit_one[random_loc:])
        else:
            new_population.append(unit_one)
            new_population.append(unit_two)
    for (index, unit) in enumerate(new_population):
        new_population[index] = decode(unit)  # 解码
    return list(set(new_population))

def mutation(population, N, mutation_p):
    # print(population, N, mutation_p)
    new_population = []
    random.seed()
    for unit in population:
        unit_code = encode(N, unit)
        p = random.random()  # 获得随机概率
        if p > (1 - mutation_p):
            random_loc = random.randint(0, N-1) 
            v = unit_code[random_loc]
            unit_code[random_loc] = '0' if v=='1' else '1'
        new_population.append(decode(unit_code))
    return list(set(new_population))


# 变量设置
generation_count = 50  # 迭代次数
N = 4  # 物体总数
n = pow(2, N)  # 种群个体总数
w = [2, 3, 1, 5]  # 每个物体重量
v = [4, 3, 2, 1]  # 每个物体价值
W = 6  # 重量阈值
population = [] 

# 初始化种群
for i in range(n):
    population.append(i)
print("Original population:",population)

# 算法开始
c = 0 # 当前迭代次数
while c < generation_count:
    print('-'*10+str(c)+'-'*10)
    
    # 种群选择
    plist = getRWSPList(population, w, v , W)  # 获得总群概率list
    new_population = []
    for i in range(n):  # 适者生存
        new_population.append(RWS(population, plist))
    new_population = list(set(new_population))
    print("After selection:",new_population)
    if len(new_population) == 1:
        population = new_population
        break
    
    # 种群交叉
    couples = getRandomCouple(len(new_population))  # 获得随机配对
    new_population = crossover(new_population, couples, 0.8, N)
    print("After crossover:", new_population)
    if len(new_population) == 1:
        population = new_population
        break
    
    # 种群变异
    new_population = mutation(new_population, N, 0.1)
    print("After mutation:"+ str(new_population))
    if len(new_population) == 1:
        population = new_population
        break
    
    population = new_population
    
    c += 1

print(population)