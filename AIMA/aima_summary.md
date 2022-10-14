AIMA

[TOC]



# 导论

## Part 1	AI的初步认识

1. **AI的发展历程**

   1. 上世纪四五十年代：AI早期，致力于探索普遍规律，有了大量AI程序
   2. 上世纪六七十年代：AI计划取消，遇到困难，停滞不前
   3. 上世纪七八十年代：思考遇到过的困难，开始求助于知识领域
   4. 上世纪九十年代 - 00年代：冬天，AI计划失败，发展缓慢，开始求助数学（eg.HMM、贝叶斯网络）
   5. 00年代 - 10年代：数学理论为ML、DL焕发生机，大数据和ML开始兴起

2. 人工智能之父

   1. 艾伦·麦席森·**图灵**——计算机逻辑奠基人
   2. 约翰·**麦卡锡**——提出“人工智能”概念
   3. 马文·明斯基——创建首个Snare
   4. 西摩尔·帕普特——将科技于教育融合

3. **三大主义及其定义的智能**

   1. **符号主义**

      “思考”表现为对物理系统和问题的符号表示，以及基于符号的推导

   2. **连接主义**

      像人一样“思考”，认为人类的思考以人脑的生物神经网络为载体

   3. **行为主义**

      恰如其分的“行动”，“行动”基于“控制论”的**感知-建模-决策-执行**模型以及对一些群体性生物行为的观察

4. **机器学习高涨原因**

   1. 理论基础的长期沉淀（前期败因）
   2. 硬件计算技术快速发展（前期败因）
   3. 大数据的兴起
   4. 现实需要及恰当的应用

## Part 2	Agent

1. **理性** = 正确的事

   ​		对于给定的信息，最大化目标实现，完全依赖于目标

2. **理性的决定因素[4]**

   1. 性能测量
   2. Agent之前的认识
   3. Agent感知时间序列
   4. 可执行的action

3. **PEAS[4]**

   1. Performance measure
   2. Environment
   3. Actuators
   4. Sensors

4. Environment 属性**[6]**

   1. 完全可观 vs 部分可观

      判定：Sensor能否检测到所有与执行Action相关的条件

   2. 确定性 vs 随机性

   3. 片段式 vs 延续式

   4. 静态 vs 动态

   5. 离散 vs 连续

   6. 单智能体 vs 多智能体

5. Agent程序结构：Agent = Architecture + Program

6. 理性的Agent

   1. Agent：使实际性能最大化
   2. **理性的Agent**：根据**目标、证据和约束**做出**最佳决策**的**系统**，与Agent相比还要**收集信息、学习**
   3. **全知的Agent**：明确知道它的动作产生的实际结果，so. 理性 $\ne$全知

7. 基于效用的Agent

   ​		仅靠目标可能不足以生成高质量的行为，用utility评估。

   ​		效用函数将（序列）状态映应到实数

8. 学习的Agent：依赖于4个执行部件**[4]**

   1. Performance element
   2. Learnning element
   3. Critic
   4. Problem generator

# 搜索

## Part 0	搜索

1. 搜索问题的形式化**[5]**
   1. 初始状态
   2. 动作
   3. 状态转移
   4. 结果判定
   5. 代价评估
   
2. 对搜索问题的性能分析**[3]**
   1. 完备性：是否有解
   2. 最优性：最低代价/最优解
   3. 复杂性：时空复杂
   
3. 关于为什么要引入搜索问题

   在AI领域中的很多问题对其求解是NP / NP complate，所以很难对其直接求解，故而只能采用搜索的方法得到解

简单搜索问题伪代码如下

```pseudocode
% 给定传感器信息 求取 对给定问题的行动序列
% 我们规定，将问题定义为简单开放闭环，即得到规划动作后无视传感器信息
function	SIMPLE_SEARCH(percept)	return an action
	var:seq		an action sequence,initailly empty
    	state	some description of currenct world
    	goal	a goal,initailly null
    	problem	a problem formulaiton
    % code
    state := UPDATE_STATE(state,percept)	% 获取当前状态
    if seq is empty then					% 队列判空
    	goal := FORMULATE_GOAL(state)		% 优先判定当前状态是否为goal
    	problem := FORMULATE_PROBLEM(state,goal)	% 根据当前状态和goal将问题形式化
    	seq := SEARCH(problem)				% 对形式化问题搜索解得到动作序列
    	if seq = failure	then return null
	action := FIRST(seq)					% 对返回结果（动作序列）赋值
	seq := REST(seq)						% 释放已规划序列
	return action
```



## Part 1	经典搜索

### 无信息搜索

基本思想：指除了问题本身以外没有其他信息

### 宽度优先搜索

基本思想：在下一层的任何节点拓展之前，搜索树上本层的所有节点应该已经被拓展过

性能分析

伪代码如下

```pseudocode
% 对于给定形式化问题，return一个解SOLUTION(node) or failure
function BREADTH_FIRST_SEARCH(problem)	return a solution or failure
	var:frontier	a node quene
		node		Search-Tree node,including state and child node
		explred		a set of expolred node(state)
		SOLUTION	a function of find solution	to given node
	% code
	node :=	problem.INITIAL_STATE()				% 根据给定problem初始化节点
	if problem.GOAL_TEST(node.STATE)	then return SOLUTION(node)	% goal判定
	frontier := INSERT(node, frontier)
	explored = null
	loop do
		if EMPTY(frontier)		then return failure
    	node := POP(frontier)						% 获取新节点
    	expolred := INSERT(node.STATE,expolred)		% 对该节点标记expolred
    	for each action in problem.ACTIONS(node.STATES) do	% 遍历所有可执行动作遍历得到子节点
     		child := CHILD_NODE(problem,node,action)		% 获取当前节点的子节点
       		if child.STATE is not in explored or frontier then
       			if problem.GOAL_TEST(child.STATE)	then return SOLUTION(child)
       			frontier := INSERT(child, frontier)			% 将该子节点放入队列中
```

### 一致代价搜索

基本思想：不再拓展最近的节点，转而每次拓展路径代价最小$g(n)$的节点$n$

特别的，当所有节点之间的路径代价都一致时，其等价于宽度优先搜索

性能分析：

伪代码如下

```pseudocode
function UNIFORM_COST_SEARCH(problem)	return solution(node) or failure
	val:frontier	a priorty quene orderd by PATH_COST,with node as the only element
	
	code:
	node := problem.INTIAL_NODE()
	frontier = INSERT(node, frontier)
	expored = null
	
	loop do
		if(EMPTY(frontier))		then return failure
		node := POP(frontier)							% 得到新节点（必为当前状态代价最低节点）
        if problem.GOAL_TEST(node)		then return SOLUTION(node)
		expored := PUSH_BACK(node.STATE)			 
		for each action in problem.ACTION(node.STATE)	do
			child := CHILD_NODE(problem,node,action)	% 得到子节信息，内置STATE,COST
            if(child.STATE)	is not in expolred or frontier then
            	frontier := INSERT(child,frontier)		% 放入优先级队列
            else if child.STATE is in frontier with higher PAHT_COST then
            	replace that frontier node with child	% 更改优先级队列顺序
```

### 深度优先搜索

基本思想

性能分析



### 深度迭代搜索

基本思想

性能分析

伪代码如下

```pseudocode
% 利用limit进行递归搜索
function DEPTH_LIMIT_SEARCH(problem,limit)	return SOLUTION(node) or false/cutoff
	return RECURSIVE_DLS(MAKE_NODE(problem.INITIAL_STATE),problem,limit)

function RECURSIVE_DLS(node,problem,limit)		return SOLUTION(node) or false/cutoff
	% code:
	node := problem.INITIAL_STATE()
	if(problem.GAOL_TEST(node))		then return SOLUTION(node)
	else if limit =  0 then return cut_off
	else
		cutoff_occureed := false
		for each action in problem.ACTIONS(node.STATE) do	% 遍历每个节点
			child := CHILD_NODE(problem,node,action)
			reuslt := RECURSIVE_DLS(child,problem,limit - 1)
			if result not is cutoff then cutoff_occurred = true
			else if result not is false	return result
        if cutoff_occurred		then return cutoff	else return false
```

---

### 有信息搜索（启发式搜索）

基本思想：利用超出问题定义本身、问题所特有的知识，找到更有效的解

一般方法：

1. 评价函数$f(n)$
2. 启发式函数$h(n)$



### 最佳优先搜索

基本思想：一个节点被选择进行拓展是基于一个评价函数$f(n)$，但大多数的最佳优先算法还包含一个启发式函数$h(n)$

实现方法：与一致代价搜索相同

区别：最佳优先搜索使用$f(n)$来代替$g(n)$来整理优先队列

启发式函数：$h(n) = $从节点$n$到目标状态的最低代价估计



### 最佳优先搜索——贪婪搜索

基本思想：优先拓展最接近目标的节点

评价函数：$f(n) = h(n)$

性能分析：b-分支因子，m-搜索空间的最大深度

1. 最差时间复杂度：$O(b^m)$

2. 空间复杂度：$O(b^m)$

   

### 最佳优先搜索——A*搜索

基本思想：避免拓展代价高的路径，使得总得到估计求解代价最小化

评价函数：$f(n) = g(n) + h(n)$

1. $g(n)$-到达该节点的代价
2. $h(n)$从该节点到目标的代价估计

定理：A*搜索时最优的

### 迭代加深A*搜索

基本思想：
		迭代加深深度优先搜索的进化版，使用启发式函数来评价到达目标的剩余代价

特点：
		因为是一种优先深度搜索，所以内存使用率低于A*
		但不同于标准迭代加深，它集中于探索最有希望的节点，因此不会去搜索树的任何处的同样深度

区别

1. 迭代加深深度优先搜索：使用搜索深度作为每次迭代的截止值
2. 迭代加深A*搜索：使用了信息更丰富的评价函数



### 递归最佳优先搜索（RBFS）

伪代码如下

```pseudocode
function RECURSIVE_BEST_FIRST_SEARCH(problem)		return SOLUTION(node) or false
	return RBFS(problem,MAKE_NODE(problem.INITIAL_STATE),\infty)

function RBFS(problem,node,f_limit)	return SOLUTION(node) or false and a new f*cost limit
	% code
	if problem.GOAL_TEST(node.STATE)	then return SOLUTION(node)
	successors := []
	for each action in problem.ACTIONS(node.STATE)	do
		add CHILD_NODE(problem,node,action)	into successors
	if successors if empty then return false.\infty
	for each s in successors do 
		s.f := max(s.g + s.h,node.f)
	loop do
		best := the lowest f-value node in successors
		if best.f > f_limit		then return false,best.f
		alternative := the second_lowest f_value among successors
		result,best.f := RBFS(problem,best,min(f.limit,alternative))
		if result is not false		then return result
```



## Part 2	超越经典搜索

### 局部搜索

条件：环境时可观察的、确定的、已知的，问题解是一个环境序列

基本思想：考虑从一个或多个状态进行评价和修改，而不是系统地探索从初始状态开始的路径

适用范围：关注解状态**而不是路径代价**的问题

缺点：非系统化

优点：

1. 很少的内存——通常为常数
2. 经常能在系统化算法不适用的很大的（连续）状态空间中找到合理的解

其他应用：解决纯粹的最优化问题，其目标是根据目标函数找到最佳状态

### 局部思想——爬山法（贪婪局部搜索）

基本思想

性能分析

存在状态

1. 局部最大值 < 全局最大值
2. 山脊：造成了一系列的局部最大值，贪婪很难处理
3. 高原（山肩）：可能是一块平的局部最大值，造成迷路

优化：随机重启爬山法——间接改变从来不会下山的现状

伪代码如下

```pseudocode
% 爬山法 （最陡上升版本）
function HILL_CLIMBING(problem)		return a state that is a local maximum
	% code
	current := MAKE_NODE(problem.INITIAL_STATE)		% 任意局部节点初始化
	loop do
		neighbor := a highest_valued successor of current	% 获取临近状态
		if neighbor.VALUE <= current.VALUE	then return current.VALUE	% 当前值为局部最大
		current := neighbor		% enter loop
```



### 局部思想——模拟退火

基本思想：允许下山的随机爬山法，并随时间推移下山时间越来越少

具体的：以评估值$\Delta E$和温度$T$表示随机移动的概率，移动的结果反作用于评估值和温度

性能评估：

1. 解的最优性：如果调度让T下降的足够慢，算法找到全局最优解的概率逼近于1

应用：

1. VLSI布局问题
2. 工厂调度
3. 其他大型最优化任务

伪代码如下

```pseudocode
% 模拟退火 
function SIMULATED_ANNEALING(problem,schedule)		returns a solution state
	val:schedule	a mapping from time to "temperature"
	% code
	current := MAKE_NODE(problem.INITIAL_STATE)		% 给定当前局部状态
	for t = 1 to \infty do		
		T := schedule(t)							% 由schedule决定当前温度
		if T = 0	then return current				% 循环终止条件
		next := a random selected successor of current	% 随机下山（或上山）
		DeltaE := next.VALUE - CURRENT.VALUE		% 评估值
		if DeltaE > 0	then current := next		% 情况变好则Next
		else current := newx only wiht probability e^{\DeataE / T}	% 情况变差则以概率(<1)接受
```

### 局部搜索——局部束搜索

基本思想：从随机K个状态开始搜索，并同时记录其后继，找到GOAL则终止

性能评估：对于最简单的局部束，最终所有的后继会聚集在一小块区域，代价比爬山法还要高昂

优化：随机局部束搜索：自然选择，根据适当的值产生合适的后继

### 局部搜索——遗传算法（随机束搜索的变形）

基本思想：以适度值函数产生评估值，以此产生后继（伴随着杂交、变异过程）

性能分析

适用条件：临近位置较为相关（因为只有很少的临近区可以受益）
遗传算法在模式具备真正与解相对应时才工作的最好

应用：

1. 电路布局
2. 作业车间调度问题

伪代码如下

```pseudocode
% 遗传算法
function REPRODUCE(x,y)		return an individual		% 杂交
	val:x,y		parent individuals
	% code
	n := length(x)
	c := random number from 1 to n
	return APPEND(SUBSTRING(x,1,c),SUBSTRING(y,c,n))

function GENETIC_ALGORITHM(population,FITNESS_FN)		return an individual
	val:FITNESS_FN		a function that measures the fitness of an individual
	% code
	while
		new_population := empty	set
		for i = 1 to SIZE(population) do
			x := RANDOM_SELECTION(population, FITNESS_FN)	
			y := RANDOM_SELECTION(populaiton, FITNESS_FN)
			child := REPRODUCE(x,y)					% 杂交
			if(samll random probability is fit enough)	then child := MUTATE(child)
			add child to new_population
		population := new _population
	until some individual is fit enough,or enough time has elasped
	return the best individual in population,according to FITNESS_FN
```

### 粒子群优化

伪代码如下

```pseudocode
% 
function PARTICLE_SWARM_OPTIMIZAITON()
	% code
	for each particle
		Initialize particle
	do 
		for each particle
			Calcululate fitness value
			if fitness value > best fitness value in history
				set fitness vlue as new pBest
			Choose particle with best fitness value of all paticles as gBest 
		for each particle
			Calculate partcle velocity
			Update particle position
	while maximun iterations or minimum error criteria is not attened
```









---

条件放开：不再强求环境的确定性和可观察性

基本思想：Agent需要对状态进行跟踪

具体的：需要考虑当传感器接收到应急情况时应该做什么

---

### 与或搜索

基本思想

性能分析

伪代码如下

```pseudocode
% 与或搜索树
function AND_OR_GRAPH_SEARCH(problem)		return a conditional plan or false
	OR_SEARCH(problem.INITIAL_STATE,problem,[])

function OR_SEARCH(state,problem,path)		return a conditional plan or false
	if problem.GOAL_TEST(state)		then return the empty plan
	if state is on path		then return false
	for each action in problem.ACTIONS(state) do
		plan := AND_SEARCH(RESULTS(state, action),,[state | path])
		if plan not is false 	then return [action | plan]
	return false 
	
function AND_SEARCH(states, problem, path)		return a contional plan or false
	for each s_i in states do
		plan_i := OR_SEARCH(s,problem,path)
		if	plan_i = false 	then return false
	return [if s_i then plan_i ...]
```



### 联机搜索

基本思想

性能分析

伪代码如下  

```pseudocode
% 
function ONLINE_DFS_AGENT(s)	return an action
	var:s	a percept that identifies the current state
	
	if GOAL_TEST(s)		then return stop
```

### 在线搜索

基本思想：对完全未知的空间从头开始搜素



## Part 3	对抗搜索

### 极大极小搜索

伪代码如下





### Alpha、Beta剪枝优化搜索

伪代码如下



## Part 4	CSP

### CSP概念引入

### 相容

结点相容

弧相容

伪代码如下

```pseudocode

```

路径相容

k相容

全局约束

### CSP回溯搜索

三个问题及其优化

###### 1. 确定选择变量的顺序和值

###### 2.推理？

###### 3.避免犯相同的错误

伪代码如下

```pseudocode

```

### CSP局部搜索

伪代码如下

```pseudocode

```

### CSP结构



# 逻辑、知识与推理

## Part 1	逻辑Agent

## Part 2	一阶逻辑

## Part 3	一阶逻辑的推理



# 规划

## Part 1	经典规划与调度问题

## Part 2	分层规划

## Part 3	多智能体规划

## Part 4	决策理论规划



# 学习
