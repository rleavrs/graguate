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







## Part 1	经典搜索

>  无信息搜索：指除了问题本身以外没有其他信息

### 宽度优先搜索

### 一致代价搜索

### 深度优先搜索

### 深度迭代搜索

> 有信息搜索

### 贪心搜索

### A*搜索

## Part 2	超越经典搜索

### 局部搜索

### 爬山法

### 模拟退火

### 局部束搜索

### 遗传算法

### 联机搜索



## Part 3	对抗搜索

### 博弈形式下的搜索问题

### Alpha、Beta算法

### 剪枝优化



## Part 4	CSP



### 回溯搜索



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

<pre class="pseudocode">
% This quicksort algorithm is extracted from Chapter 7, Introduction to Algorithms (3rd edition)
\begin{algorithm}
\caption{Quicksort}
\begin{algorithmic}
\PROCEDURE{Quicksort}{$A, p, r$}
    \IF{$p < r$} 
        \STATE $q = $ \CALL{Partition}{$A, p, r$}
        \STATE \CALL{Quicksort}{$A, p, q - 1$}
        \STATE \CALL{Quicksort}{$A, q + 1, r$}
    \ENDIF
\ENDPROCEDURE
\PROCEDURE{Partition}{$A, p, r$}
    \STATE $x = A[r]$
    \STATE $i = p - 1$
    \FOR{$j = p$ \TO $r - 1$}
        \IF{$A[j] < x$}
            \STATE $i = i + 1$
            \STATE exchange
            $A[i]$ with $A[j]$
        \ENDIF
        \STATE exchange $A[i]$ with $A[r]$
    \ENDFOR
\ENDPROCEDURE
\end{algorithmic}
\end{algorithm}
</pre>