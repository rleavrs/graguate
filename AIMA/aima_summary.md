AIMA

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
   2. 最优性：最低代价
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

>  无信息搜索：指除了问题本身以外没有其他信息

### 宽度优先搜索

基本思想：在下一层的任何节点拓展之前，搜索树上本层的所有节点应该已经被拓展过

性能分析：

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

> 有信息搜索

### A*搜索

基本思想

性能分析

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

基本思想

性能分析

伪代码如下

```pseudocode

```

### 爬山法

基本思想

性能分析

伪代码如下

```pseudocode

```



### 模拟退火

基本思想

性能分析

伪代码如下

```pseudocode

```

局部束搜索

### 遗传算法

基本思想

性能分析

伪代码如下

```pseudocode

```



### 与或搜索





### 联机搜索

基本思想

性能分析

伪代码如下

```pseudocode

```



## Part 3	对抗搜索

### 极大极小搜索

伪代码如下





### Alpha、Beta剪枝优化搜索

伪代码如下



## Part 4	CSP

### 弧相容算法搜索 -AC3



### CSP简单回溯搜索



### 结构CSP



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
