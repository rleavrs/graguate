# 导论 Part1

## Smmary

Four categories of AI goals: Thinking Humanly, Thinking Rationally, Acting Humanly and Acting Rationally

AI的分类可分为4种：类人思考, 类人动作,理性思考和理性动作。



Eight Foundations of AI: Philosophy, Mathematics, Economics, Neuroscience, Psychology, Computer science, Control theory and cybernetics, and Linguistics.

8个基础学科包括：哲学、数学、经济学、神经科学、心理学、计算机工程、控制理论和控制论和语言学。



Three types of AI: Weak AI, Strong AI and Super AI

AI的3种类型：弱人工智能、强人工智能以及超人工智能。



# 导论 Part2

讨论：Learned the six approaches for artificial intelligence, do you think which one could be better? Why?

学习了人工智能的六种途径，你认为哪种途径更好？为什么？



智能Agent范式可以看做一个大框架，用于研究智能机器人的总体设计。而统计方法是一个有力工具，可以渗透学习算法的方方面面。至于学习算法的具体形式，或许会是目前流行的联结主义，也可能会是一种新形式的符号主义，又或者是两者的结合。大脑仿真和逻辑推理也是一样，两种方法各有优势，如果能够结合起来说不定也会迸发出新的火花。总之，目前我们还处于人工智能发展的探索阶段，谁也不知道最终哪条路才是正确。因此我认为这六种途径并不是绝对对立的，而是相辅相成的。每一个想法都值得研究，每一种途径都值得尝试。



我认为这六种途径并不是绝对对立的，而是相辅相成的。智能Agent范式可以看做一个大框架，用于研究智能机器人的总体设计。而统计方法是一个有力工具，可以渗透学习算法的方方面面。至于学习算法的具体形式，或许会是目前流行的联结主义，也可能会是一种新形式的符号主义，又或者是两者的结合。大脑仿真和逻辑推理也是一样，两种方法各有优势，如果能够结合起来说不定也会迸发出新的火花。总之，目前我们还处于人工智能发展的探索阶段，谁也不知道最终哪条路才是正确。因此我认为，每一个想法都值得研究，每一种途径都值得尝试





讨论：For the five typical classes of agents, what are the differences between them? And can you design different type of agent?

对于五种主要的智能体，他们之间的差异是什么？你能设计一种不同类型的智能体吗？



五种智能体的差异在于他们各自解决问题手段的智能性和效率之间增加。简单反射智能体仅在当前感知的基础上动作，忽略其余的感知历史；基于模型反射的智能体需要依赖于感知的历史，能够反射某些当前状态无法观测的方面；基于目标的智能体能够在多个可能性之间选择一种方式，挑选出达到目标状态的那一个；基于效用的智能体根据智能体的“高兴”程度，允许对不同的外部环境状态进行比较；学习智能体利用评论者对智能体如何动作的反馈，然后决定如何修改性能要素以便未来做得更好。其他智能体比如决策智能体，输入智能体，处理智能体，空间智能体，时间智能体，世界智能体，可信智能体，购物智能体，客户服务台，个人智能体和数据挖掘智能体





## Summary

AI的主要方法：控制论与人脑仿真、符号与亚符号、基于逻辑与反逻辑、符号主义与联结主义、统计学、以及智能体范例。

一个智能体是感知并作用于外部环境的任何事物。

典型的智能体：简单反射智能体、基于模型的反射智能体、基于目标的智能体、基于效用的智能体、以及学习智能体。



# 搜索 Part1

### 讨论：什么是无信息搜索？

What is uninformed search? How many search strategies used for it? And what are the differences between those strategies?

什么是无信息搜索？其搜索策略有多少种？并且这些策略之间的差异是什么？



### 讨论：什么是有信息搜索？

What is informed search? What is the heuristic function? What is the relation between the informed search and the heuristic function.

什么是有信息搜索？什么是启发式函数？有信息搜索与启发式函数之间的关系是什么？



## Summary

Solving a problem is a sequence of actions, and search is the process of looking for the actions to reach the goal.

求解一个问题就是一系列动作，并且搜索是为达到目标寻找这些动作的过程。



Uninformed search is also called blind search: the typical algorithms are Breadth-first, Uniform-cost, and Depth-first.

无信息搜索亦被称为盲目搜索：其代表性算法是宽度优先、一致代价、以及深度优先。



Informed search is also known as heuristic search: Best-first search is according to an evaluation function, Its special cases are Greedy Search and A* search.

有信息搜索也被称为启发式搜索：最佳优先搜索依赖于评价函数，其特例是贪婪搜索和A*搜索。



Time and space complexity are some key points for a search algorithm.

时间和空间复杂性是搜索算法的一些关键点。







# 搜索 Part2

### 讨论：如何理解局部搜索？

What is local search? And how is it different than classical search?

什么是局部搜索？ 它与经典搜索有什么不同？

局部搜索：是指能够无穷接近最优解的能力，而全局收敛能力是指找到全局最优解所在大致位置的能力，局部搜索能力与全局搜索能力，缺一不可，向最优解的导向对于任何智能算法的性能都是很重要的。





### 讨论：优化算法和群体智能之间的关系

From the viewpoint of optimization problems, is there any relation between optimization algorithms and swarm intelligence? Why?

从优化问题的角度来看，优化算法与群体智能之间有什么关系吗？为什么？

优化算法和群体智能都是根据已知信息计算后续可能状态，下一状态从后续可能状态中以某个函数计算得出。 区别在于，优化算法有单个或少量初始节点，而群体智能有大量初始节点\



## Summary

Local search: Hill-Climbing operate on complete-state formulations; Local Beam keeps track of k states; Tabu Search uses a neighborhood search with some constraints.

局部搜索：爬山法在完整状态形式化上进行操作；局部束搜索法保持k个状态的轨迹；禁忌搜索采用一种带约束的邻域搜索。



Optimization and Evolutionary Algorithms: Simulated Annealing approximate global optimization in a large search space; Genetic Algorithm mimics the evolutional process of natural selection.

优化与进化算法：模拟退火在大搜索空间逼近全局最优解；遗传算法模仿自然选择的进化过程。



Swarm Intelligence: Ant Colony Optimization can be reduced to finding good paths through graphs; Particle Swarm Optimization is by iteratively trying to improve a candidate solution.

群体智能：蚁群优化可以寻找图的最好路径；粒子群优化通过迭代来改善一个候选解。



# 搜索 Part3

### 讨论：哪种类型的游戏难以用人工智能来实现？

For the two types of games, i.e. perfect information (fully observable) and imperfect information (partially observable), which type is harder to be implemented by artificial intelligence? Why? 

对于这两种类型的游戏，即完美信息（完全可观察）和不完美信息（部分可观察），哪种类型难以用人工智能来实现？为什么？

不完美信息（部分可观察）这种类型难以用人工智能来实现。因为其充满了随机性，能确定的只有出现的和自己知道的，这样的条件通过算法去推算对方手中的牌或者信息，首先是需要很多时间与内存去实现；其次，每次出牌或出现新的信息后又要进行重新推导，又需要花费时间与内存；再而就是衡量的标准很难确定，因为推导至对方手牌或信息有概率成分，从而加大算法的难度，得出结果的时间与占用的空间。



### 讨论：什么是蒙特卡洛方法？

What are Monte-Carlo methods? Why is it able to be used for Computer Go?

什么是蒙特卡洛方法？为什么可以被用于计算机围棋？

蒙特·卡罗方法（Monte Carlo method），也称统计模拟方法，是二十世纪四十年代中期由于科学技术的发展和电子计算机的发明，而被提出的一种以概率统计理论为指导的一类非常重要的数值计算方法。是指使用随机数（或更常见的伪随机数）来解决很多计算问题的方法。与它对应的是确定性算法。将蒙特卡洛方法用于计算机围棋，能够与价值和策略网络结合，使得评价棋盘位置和选择具体走棋方法往最有利的方向发展。因为在计算机围棋中蒙特卡洛方法所对应的概率分布可由计算机存储的先验概率分布决定，因此这也就使得结果朝着更好的方向发展。



## Summary

Minimax algorithm can select optimal moves by a depth-first enumeration of game tree.

Minimax算法可以通过博弈树的深度优先计算选择最佳移动。



Alpha–beta algorithm achieves much greater efficiency by pruning irrelevant subtrees.

Alpha–beta算法通过剪掉不相关子树来得到更高的效率。



Heuristic evaluation function is useful for imperfect real-time decisions of games.

启发式评价函数对于博弈的不完全实时决策很有效。



Stochastic game is a dynamic game with probabilistic transitions.

随机博弈是具有概率转换的动态博弈。



Monte-Carlo tree search combines Monte-Carlo simulation with game tree search. 

蒙特卡罗树搜索将蒙蒙特卡罗树仿真与博弈树搜索相结合。

# 搜索 Part4

### 讨论：如何理解约束满足问题？

What are Constraint Satisfaction Problems? And how are they different from Standard Search?

什么是约束满足问题？ 它与经典搜索有什么不同？

约束满足问题由一个变量集合和一个约束集合组成。问题的一个状态是由对一些或全部变量的一个赋值定义的完全赋值：每个变量都参与的赋值。问题的解是满足所有约束的完全赋值，或更进一步，使目标函数最大化。 它与标准搜索相比受到约束条件限制。

由一个变量集合和一个约束集合组成。问题的一个状态是由对一些或全部变量的一个赋值定义的完全赋值：每个变量都参与的赋值。问题的解是满足所有约束的完全赋值，或更进一步，使目标函数最大化。与标准搜索相比受到约束条件限制



### 讨论：澳大利亚地图着色问题有多少解？

How many solutions are there for the map-coloring problem of Australia? And list all of those solutions.

对于澳大利亚地图着色问题，有多少个解？并且列举所有这些解。

澳大利亚地图着色问题应该是采取3种颜色的，分别定为red（r）, green（g）, blue（b）。

{WA, NT, SA, Q, NSW, V, T}的满足约束解共有18种，情况如下：

{r, g, b, r, g, r, r} {r, g, b, r, g, r, g} {r, g, b, r, g, r, b} 

{r, b, g, r, b, r, r} {r, b, g, r, b, r, g} {r, b, g, r, b, r, b}

{b, g, r, b, g, b, r} {b, g, r, b, g, b, g} {b, g, r, b, g, b, b}

{b, r, g, b, r, b, r} {b, r, g, b, r, b, g} {b, r, g, b, r, b, b} 

{g, r, b, g, r, g, r} {g, r, b, g, r, g, g} {g, r, b, g, r, g, b} 

{g, b, r, g, b, g, r} {g, b, r, g, b, g, g} {g, b, r, g, b, g, b} 



## Summary

CSPs represent a state with a set of variable/value pairs and represent the conditions by a set of constraints on the variables.

CSPs问题用一组变量/值对表示状态，并且用一组变量的约束表示条件。



Node, arc, path, and k-consistency use constraints to infer which variable/value pairs are consistent. 

节点、弧、路径、以及k一致性使用约束来推断哪个变量/值对是一致的。



Backtracking search, and local search using min-conflicts heuristic are applied to CSPs.

回溯搜索以及采用最少冲突启发式的局部搜索被用于CSPs。



Cutset conditioning and tree decomposition can be used to reduce a constraint graph to a tree structure.

割集调节和树分解可被用于将约束图简化为树结构。



# 推理



### 讨论：命题逻辑和一阶逻辑的差异

What are the differences between Propositional Logic and First-order Logic?

命题逻辑与一阶逻辑之间的差异是什么？

题逻辑处理的最小单位是命题，一阶逻辑则可以进一步分析语句中的成分。比如说，命题逻辑只能判定整句句子的真假。但是，像当今法国国王是秃子。这样的命题就超出了命题逻辑的判断范围，因为它既不是真命题，也不是假命题。（当今法国国王不是秃子也是假命题，因为根本不存在法国国王。）于是，我们就引入了存在这个符号。在一阶逻辑中，是可以进行判断的。

### 讨论：为什么贝叶斯规则对于不确定知识的推理很有用？

Why Bayes’ Rule is useful for the reasoning on uncertain knowledge? Give an instance to explain this question.

为什么贝叶斯规则对于不确定知识的推理很有用？ 给出一个实例来解释这个问题。

已知某种疾病的发病率是0.001，即1000人中会有1个人得病。现有一种试剂可以检验患者是否得病，它的准确率是0.99，即在患者确实得病的情况下，它有99%的可能呈现阳性。它的误报率是5%，即在患者没有得病的情况下，它有5%的可能呈现阳性。现有一个病人的检验结果为阳性，请问他确实得病的可能性有多大？假定A事件表示得病，那么P(A)为0.001。这就是"先验概率"，即没有做试验之前，我们预计的发病率。再假定B事件表示阳性，那么要计算的就是P(A|B)。这就是"后验概率"，即做了试验以后，对发病率的估计。根据条件概率公式，用全概率公式改写分母，将数字代入，我们得到了一个惊人的结果，P(A|B)约等于0.019。也就是说，即使检验呈现阳性，病人得病的概率，也只是从0.1%增加到了2%左右。这就是所谓的"假阳性"，即阳性结果完全不足以说明病人得病





## Summary

Knowledge representation captures information. Its typical methods are semantic network, first order logic, production system, ontology and Bayesian network.

知识表示捕捉信息。其代表性的方法是：语义网络、一阶逻辑、产生式系统、本体和贝叶斯网络。

Ontological engineering is to study the methods and methodologies for building ontologies.

本体工程是研究构建本体的方法和方法学。

Uncertain knowledge can be handled by probability theory, utility theory and decision theory. 

不确定性知识可以用概率论、效用论和决策论来处理。

Bayesian networks can represent essentially any full joint probability distribution and in many cases can do so very concisely.

贝叶斯网络基本上可以表示任意的全联合概率分布，并且在许多情况下可以做的非常简洁。

# 规划

### 讨论：经典规划和现实世界规划的差异

What is the planning problem? What are the differences between Classic Planning and Real-world Planning?

什么是规划问题？经典规划与现实世界规划之间的差异是什么？



规划意味着制定一套行动计划来达到既定的目标，经典规划的特点是完全可观测，唯一的已知初始状态，静态环境，确定性动作，每次仅有一个动作，单一智能体等，经典规划可以表示做什么，按照什么顺序，但是不能表示动作持续多长时间，什么时候发生，经典规划过于理想化，很多现实问题无法用经典规划加以解决。现实世界规划更为复杂，需要对表示语言和与外部环境交互方式上进行扩展



### 讨论：值迭代和策略迭代的区别

In dynamic planning, there are two optimal policies called Value Iteration and Policy Iteration, what are their differences?

在动态规划中，有两种称为值迭代和策略迭代的最优策略，它们有什么区别？

值迭代：计算每个状态的效用，然后使用该状态效用在每个状态中选择一个最佳动作。不使用π函数；而π值在U(s)中计算。策略迭代：交替执行策略迭代（给定一个策略πi，如果πi被执行的话，计算每个状态的效用Ui）和策略改善（使用基于Ui的提前看一步法，计算一个新的MEU（最大期待效用）策略πi+1）。 区别:前者直接选择最佳动作，后者在动作中不断完善到最佳。



## Summary

Classical planning is the simplest planning. 

经典规划是最简单的规划。



Planning graph, Boolean satisfiability, first-order logical deduction, constraint satisfaction, and plan refinement can be used.

可使用规划图、布尔可满足性、一阶逻辑推理、约束满足, 和规划精进方法。



Planning and acting in the real world are more complex. 

现实世界的规划与动作更为复杂。



The representation language and the way of agent interacts with environment should be extended.

应当扩展表示语言、以及智能体与外部环境交互的方式。



For a problem of decision-theoretic planning, Markov Decision Process and dynamic programming can be used to formulate and solve it.

对于决策理论规划问题，可使用马尔科夫决策过程和动态规划对其进行形式化和求解。





# 学习 Part1

### What is Machine Learning?

What is Machine Learning? What is Deep Learning? What are the relations among Machine Learning, Deep Learning and Artificial Intelligence?

什么是机器学习？什么是深度学习？机器学习、深度学习以及人工智能之间的关系是什么？



机器学习：机器学习是人工智能的一个分支，从事构建可以从数据中学习的系统。从数据中积累或者计算的经验获取技能。深度学习:机器学习研究中的一个新的领域，其动机在于建立、模拟人脑进行分析学习的神经网络，它模仿人脑的机制来解释数据，例如图像，声音和文本。人工智能：研究感知外部环境并为某个目标采取行动的智能体。



### 你是否同意本课中提出的关于机器学习的三个视角？

Do you agree with the three perspectives on machine learning proposed in this course? Illustrate the reason why do you agree, or why do you not.

你同意本课程提出的机器学习的三个观点吗？说明你为什么同意、或者你为什么不同意的理由。



我同意老师在本章中提出的三个视角。学习任务、学习范式、学习模型这三个视角分别从不同的角度对机器学习算法进行了分类总结，每一个视角都各有侧重，分别分析机器学习算法的不同方面。三个视角相辅相成，很全面地对现有的机器学习算法进行了归纳和总结，具有很好的指导作用。



## Summary

Machine learning is to study some algorithms that can learn from and make predictions on data. 

机器学习是研究一些可以从数据中学习、并对数据进行预测的算法。



The different perspectives are aimed to try to have a taxonomy on the algorithms of machine learning, for being easy to understand machine learning. 

几个不同视角旨在尝试对机器学习的算法进行分类，以便于理解机器学习。



Three perspectives on machine learning are proposed in this chapter, those are learning tasks, learning paradigms and learning models.

本章提出了机器学习的三个视角，他们是：学习任务、学习范例以及学习模型。



# 学习 Part2

### 头脑风暴：机器学习典型的学习任务

This chapter has listed seven typical learning tasks, i.e. Classification, Regression, Clustering, Ranking, Density estimation, Dimensionality reduction, and Optimization. Are there some learning tasks in the list do you not agree, or are there some learning tasks that have not been listed in? Why?

本章列出了七个典型的学习任务，即分类，回归，聚类，排名，密度估计，降维和优化。列表中有没有一些学习任务您不同意，或是有一些尚未列出的学习任务？为什么？

概念学习就是把具有共同属性的事物集合在一起并冠以一个名称，把不具有此类属性的事物排除出去。研究这类学习任务有助于让机器具有像人一样的对各种概念进行归纳分析的能力。



### 分类与回归之间、以及分类与聚类之间的相似性和差异是什么？

What are the similarity and the difference between classification and regression, and between classification and clustering?

分类与回归之间、以及分类与聚类之间的相似性和差异是什么？



分类：已知存在哪些类，即对于目标数据库中存在哪些类是知道的，要做的就是将每一条记录分别属于哪一类标记出来。 回归：反映了数据库中数据的属性值的特性，通过函数表达数据映射的关系来发现属性值之间的依赖关系。它可以应用到对数据序列的预测及相关关系的研究中去。在市场营销中，回归分析可以被应用到各个方面。如通过对本季度销售的回归分析，对下一季度的销售趋势作出预测并做出针对性的营销改变。 聚类：将已给定的若干无标记的模式聚集起来使之成为有意义的聚类，聚类是在预先不知道目标数据库到底有多少类的情况下，希望将所有的记录组成不同的类或者说聚类，并且使得在这种分类情况下，以某种度量（例如：距离）为标准的相似性，在同一聚类之间最小化，而在不同聚类之间最大化。



## Summary


The learning tasks are the general problems that can be solved with machine learning, and each task can be achieved by various algorithms but not a specific one.

学习的任务是可以用机器学习求解的一些通用性问题，并且每个任务可以用不同的算法来实现，而不是特定的一个。



The typical tasks in machine learning include: Classification, Regression, Clustering, Ranking, Density estimation, and Dimensionality Reduction.

机器学习中的代表性任务包括：分类、回归、聚类、排名、密度估计、以及降维。

# 学习 Part3

### 为什么将本章介绍的三种学习范式界定为机器学习的典型范式？

What is the difference between supervised learning and unsupervised learning? And what is that between unsupervised learning and reinforcement learning? Why are those three learning paradigms thought to be typical paradigms of machine learning?

有监督学习与无监督学习之间的差异是什么？而无监督学习与强化学习之间呢？为什么这三种学习范式被认为是机器学习的典型范式？



有监督学习和无监督学习的差异在于：有监督学习训练模型时存在标注数据，为训练样本提供了输入和输出对，使得模型能够从样本和标签中学习；而无监督学习所训练的数据没有标注，只能利用一定的规则发现其中隐藏的结构。无监督学习与强化学习之间的差异在于：强化学习专注于在线决策的效果，存在奖励机制，而无监督学习不能做到这样的效果。这三种作为机器学习的典型范式原因在于：三者能够较好地适用于不同的机器学习典型场景，基本上目前的大部分机器学习能够解决的任务都是可以从这三种范式中找到合适的方法解决。



### 如何看待卷积神经网络的创始人对三种学习范式的观点？

Yann LeCun, the founding father of convolutional neural network (CNN), has such a comment that “If intelligence was a cake, unsupervised learning would be the cake, supervised learning would be the icing on the cake, and reinforcement learning would be the cherry on the cake.” What is your opinion about his comment?

卷积神经网络的创始人雅恩·勒昆有过这样一段点评，“如果智能是一块蛋糕，无监督学习就是这块蛋糕，有监督学习则是蛋糕上的糖霜，而强化学习是蛋糕上的樱桃。”你对他的点评有什么看法？

作者认为有监督学习是大家都能看到的，很多基于有监督学习的算法研究已经成熟，强化学习也是如此。但无监督学习目前还没有被研究透彻，还有许多问题没有被解决，未来可能许多问题都是需要基于无监督学习来解决的，因此其发展前景更为广阔。



## Summary

机器学习的范式用于区分机器学习中不同的原型。一个学习范式就是刻画一种学习的原型，基于对学习的经验、或者与环境的交互。

对范式的类别进行研究的意义在于，为了使学习任务能够达到好的效果，会使你思考去选择一个合适的范式；反过来，了解了学习范式的类别之后，会加深对学习任务的理解。

这一章我们着重学习了三种代表性的范式：

 \- 有监督学习：是一种“样本学习（learning-by-examples）“的范式

 \- 无监督学习：是一种“自我学习（learning-by-itself）”的范式

 \- 强化学习：是一种“在线学习（online-learning）”的范式。

此外，还简单介绍了其它几种范式：

 \- Ensemble（集成式）：将大量的弱学习器组成一个强学习器

 \- Learning to learn（学会学习）：基于先前的经验学会自身的归纳性偏向。

 \- Transfer（迁移式）：关注于已学习的知识并将其用于不同但相关的问题

 \- Adversarial learning（对抗式）：以一种对抗性方式（即零和博弈）来生成满足某种分布的数据。

 \- Collaborative learning（协同式）：以一种非对抗的协同方式（即双赢）来获取所期待的结果

# 学习 Part4



### 为什么深度神经网络比浅层的好？

Why do deep neural networks work better than shallow ones? Give an instance.

为什么深度神经网络比浅层的好？请举例说明。

浅层结构算法：其局限性在于有限样本和计算单元情况下对复杂函数的表示能力有限，针对复杂分类问题其泛化能力受到一定制约。深度学习可通过学习一种深层非线性网络结构，实现复杂函数逼近，表征输入数据分布式表示，并展现了强大的从少数样本集中学习数据集本质特征的能力。

### 你如何看待“胶囊网络”？

Convolutional neural networks (CNNs) have successfully been used to image recognition, video analysis, natural language processing and so on. However, Geoffrey Hinton proposed a “capsules network”, i.e. a "capsule" is a group of neurons within a layer in the network. How do you think about this “capsule network”? Which kind of network would be better by your opinion?

卷积神经网络（CNNs）已经被成功地用于图像识别、视频分析、自然语言理解等领域。然而，杰弗里·辛顿提出了“胶囊网络”，一个“胶囊”是位于网络中某个层次的一组神经元。你如何看待这种“胶囊网络”？你认为哪个网络会更好？

浅层结构算法：其局限性在于有限样本和计算单元情况下对复杂函数的表示能力有限，针对复杂分类问题其泛化能力受到一定制约。深度学习可通过学习一种深层非线性网络结构，实现复杂函数逼近，表征输入数据分布式表示，并展现了强大的从少数样本集中学习数据集本质特征的能力。



## Summary

Learning models are such ones that an agent can choice an appropriate model from which to fulfill a learning task.

学习模型指的是这样一些模型，智能体可以选用某种合适的模型来实现某种学习任务。



Four typical types of learning models we have discussed:

本章讨论了四种典型的学习模型：

- Geometric models 几何模型
- Logical models 逻辑模型
- Networked models 网络化模型
- Probabilistic models 概率模型