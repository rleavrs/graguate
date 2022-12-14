###### 搜索

概念题：什么是搜索
A：利用已有的知识和经验，根据问题的实际情况，不断寻找可利用的知识，从而构造一条**代价最小的推理路线**，使问题得到解决的过程称为搜索

概念题：问题求解智能体解决问题的步骤为
A：目标形式化——问题形式化——搜索求解——执行动作

概念题：对问题进行形式化
A：转换模型；动作

概念题：____或____只能通过搜索求解
A：NP或NP难问题

概念题：问题形式化是给定一个目标，决定要考虑的____与____的处理
A：动作与状态

应用题：八皇后问题形式化，初始八个皇后都在棋盘上，这种形式化属于
A：全态形式化

概念题：以下不是搜索问题一般具有的特征是
A：问题一定是有解的

概念题：普通搜索问题是指
A：求出一条从初始状态到目标状态之间的行动序列

概念题：问题求解的目的包括
A：希望及其找到问题的最优解
		希望及其找到问题的一个解

---

应用题：设一个所处环境为A，B两个格子的真空吸尘器，它能执行左右移动、吸尘、休息，它能感知所处位置以及所处位置脏还是不脏。它的当前状态为（B，脏），已知此时A也脏，则它的后续状态有
A：A,脏；B,干净；B,脏

概念题：无信息搜索	
A：深度优先；广度优先；一致代价；双向搜索

概念题：树搜索算法中从待拓展节点表中选择节点进行拓展是根据？？来选择的
A：搜索策略

概念题：生成新的节点时，需要存储五种主要信息，分别是：结点的父亲结点，父亲结点执行什么____产生的，结点对应的____，结点所在____，从初始结点到达此结点的____。
A：动作、状态、深度、代价

概念题：宽度优先搜索算法的fringe表采用____队列来实现
A：FIFO 先进先出

概念题：一致代价搜索在____相等时与宽度优先搜索是一样的
A：**单步**代价

应用题：初始结点为A，A的儿子结点是B和C，B的儿子结点是D和E，C的儿子结点是F和G，D的儿子结点是H和I，则按照深度优先搜索策略，这些结点的访问顺序是
A：ABDHIECFG

判断题：深度有限搜索具有完备性。
A：×

应用题：初始结点A的儿子结点是B和C，B的儿子结点是D和E，C的儿子结点是F和G，D的儿子结点是H和I，E的儿子结点是J和K。其中F和K都对应着问题的解。请问迭代深入搜索返回的解路径是
A：ACF

概念题：迭代深入搜索具备如下性能
A：最优性；完备性；线性空间复杂度

概念题：无信息搜索中有完备性的是
A：宽度优先搜索；迭代深入搜索；代价一致搜索

概念题：盲目搜索包括
A：深度、广度优先搜索

---

概念题：使用特定知识的搜索
A：有信息搜索；启发式搜索

概念题：启发式搜索中，评价函数的作用是
A：从当前节点出发来选择后继节点

概念题：启发式搜索中，启发函数的作用是
A：计算从当前节点到目标节点之间的最小代价值

概念题：为了保证A*算法是最优的，需要启发式具有可容和一致的特点，下面对这一特点解释正确的是
A：启发函数不会过高估计从当前节点到目标节点之间的实际开销代价

概念题：如果问题存在最优解，____必然可以得到最优解
A：宽度优先搜索

概念题：从初始节点 S0 开始逐层向下扩展，在全部搜索完第 k 层节点之后，才进入第 k+1 层节点 进行搜索。这种搜索策略属于___
A：宽度优先搜索

概念题：如果问题存在最优解，____可以认为是“智能程度相对比较高”的算法
A：启发式搜索

概念题：启发性信息包括___有关的信息
A：与具体问题求解过程；知道搜索过程；搜索有希望的方向

概念题：启发性信息的作用强度会影响搜索结果，当___，可降低搜索工作量，单可能导致找不到最优解
A：强度为0

概念题：在启发式搜索策略中，closed表用于____
A：存放已经拓展过的节点

概念题：对$g(x)$和$f(x)$描述正确的是
A：$h(x)$是从节点x到目标节点的最优路径的估计代价
		$g(x)$是从初始节点到节点x的实际代价

概念题：贪婪搜索评价函数
A：$f(n) = h(n)$只有启发式，没有代价

概念题：贪心算法正确的是
A：贪心算法可以快速地找到一个可行解，但不一定找到最优解
		贪心算法也是先将一个问题分成几个步骤操作
		贪心算法在每一步选择当前看起来是最佳的选择

概念题：*搜索评价函数
A：$f(n) = h(n) + g(n)$

应用题：初始结点到达C结点的代价是120，C结点到达它的儿子结点D的单步代价是2，结点D到达目标的估计代价是60，则结点D的评估函数值是多少？____
A：182

应用题：两个可采纳的启发式函数h1(n)和h2(n)， 如果对于任意的结点n都有h1(n)大于h2(n)，则启发式函数h1比h2占优势，即h1比h2更好。
A：√

概念题：可采纳的启发式函数是从来不会____实际代价值。
A：过估计

概念题：如果A星搜索采用____启发式函数，则它能保证最优性。
A：可采纳的

概念题：下面关于启发式搜索正确的是
A：在启发式搜索中，对节点的评价是十分重要的，评价函数是搜索成败的关键
		启发式搜索可以省略大量无谓的搜索路径
		在搜索过程中对待扩展的每一个节点进行评估，得到最好的位置，再从这个位置进行搜索直到目标
		启发式搜索，也称为有信息搜索或知情搜索，借助问题的特定知识来帮助选择搜索方向

概念题：一定能够找到最优解的启发函数是包括
A：分支定界法 A*搜索算法

---

概念题：在内存中仅保存一个节点似乎是对内存限制问题的极端反应。局部束搜索保持
A：k个状态而不仅仅为一

概念题：‏遗传算法是随机束搜索的一个变体，其中后继节点的生成是由
A：组合两个双亲状态而不是修改单一状态
F：组合两个双亲节点而不是修改单一节点

应用题：在爬山搜索算法中，如果当前状态是A，A的评估函数值是16，A的后续状态有B，C，D，它们的评估值分别是6,12,18，则下一状态会是
A：D

概念题：局部束搜索与爬山搜索算法的最根本区别在于：前者同时保存多个状态，而后者任何时候 都只保存一个状态。
A：√

概念题：‌爬山搜索有时也被称为贪婪局部搜索，因为它只顾抓住一个好的邻接点的状态，而不提前思考下一步该去哪儿。它在三种情况下经常被困
A：高原；局部最大值；山岭

概念题：爬山法可能出现的问题有
A：高原问题、山脊问题、山麓问题

概念题：爬山法
A：爬山法是一种贪心算法
		爬山法是一种局部择优的方法
		爬山法没有能力从错误中或错误路径中恢复
		爬山发需要保存未选择路径的记录（X）

概念题：‌以下关于模拟退火算法的陈述哪些是正确的？
A：模拟退火算法不是选择最佳行动，而是选择随机行动。
		模拟退火算法的内循环与爬山法非常相似。

概念题：蚁群优化算法是受蚂蚁在_______和食物源之间寻找路径行为的启发而形成的。
A：蚁巢

概念题：‎受鸟类和鱼类的社会行为的启发，粒子群优化算法采用若干_______构成一个围绕搜索空间移动的群体来寻找最优解
A：粒子

概念题：‍局部搜索算法使用一个______（而不是多条路径），并且通常仅移动到该节点相邻的节点。
A：当前节点

填空题：‏除了寻找目标之外，局部搜索算法对解决纯_________也很有效。其目的是根据一个目标函数找到其最好的状态。
A：优化问题

概念题：遗传算法主要借用生物进化中的____的规律
A：适者生存

概念题：遗传算法的适应度函数是____的标准
A：用来区分群体中的个体好坏

概念题：遗传算法中起核心作用的是____
A：交叉算子

概念题：遗传算法采用群体搜索策略，同时对搜索空间中的多个解进行评估，因此
A：遗传算法具有较好的全局搜索性能

性质：遗传算法不能保证每次得到最优解

概念：
		生物进化过程中选择遗传和变异起作用，同时又使变异和遗传向这适应环境方向发展
		选择是通过遗传和变异起作用的，**变异为选择提供资料，遗传巩固与积累选择的资料**，而选择能控制变异与遗传的方向，使变异和遗传向着适应环境的方向发展

概念题：遗传算法中，将所有妨碍适应度高的个体产生，从而影响遗传算法正常工作的问题统称为____
A：欺骗问题

概念题：在遗传算法中，适应度函数的设计要结合问题本身的要求而定，但
A：适应度函数和问题的目标函数没有关系

概念题：在遗传算法中，___，但不是说一定都能够被选上
A：适应度大的个体被选择的概率大

概念题：关于蚁群算法，描述正确的是
A：蚁群算法是通过人工模拟蚂蚁搜索食物的过程，即通过个体之间的信息交流和互相协作，最终找到从蚁穴到食物源的最短路径的
		蚁群算法是一种应用于**组合优化问题的启发式**搜索算法
		蚂蚁系统是一种增强型学习系统

概念题：蚁群算法的参数
A：信息素启发因子越小，一群搜索的随机性越小
		信息素启发因子越大，蚂蚁选择以前走过的路径的可能性越大，蚁群的搜索过程越不易陷入局部最优

概念：蚂蚁在运动过程中，根据各条路径的信息素决定转移方向



---

概念题：博弈的概念
A：对抗搜索通常称为博弈

概念题：___智能体交互总损失可以不为0
A：非零和博弈

概念题：‌从如下关于零和博弈maximum概念中选择正确的答案。
A：每个玩家会使对手可能的最大损失变得最大。
		每个玩家会使自己可能的最大收益变得最大。

概念题：‌以下关于alpha–beta剪枝的陈述哪些是正确的？
A：Alpha–beta剪枝旨在消除其搜索树中由minimax算法评价的大部分。
		Alpha–beta剪枝旨在减少其搜索树中由minimax算法评价的节点数量。

概念题：克劳德·香农提出：程序应该早一些剪断搜索，并在搜索中对状态应用____________评价函数，有效地将非终端节点转换为终端叶节点。
A：启发式

概念题：____________是一种具有概率转换的动态博弈，有一个或多个玩家。（请填写中文答案）
A：随机博弈

概念题：‌蒙特卡罗方法是一大类计算算法，它凭借________________来获得数值结果。（请填写中文答案）
A：重复随机采样

概念题：‍___________树搜索对最有利的动作进行分析，根据搜索空间的随机采样来扩展搜索树。（请填写中文答案）
A：蒙特卡罗

应用题： 智能体调用极小极大值算法搜索发现，如果它执行a动作得到的分值为10，执行b动作得分8，执行c动作得分8，执行d动作得分12，则它会执行什么动作？
A：D

判断题：效能函数通常是针对max结点，即智能体来定义的，所以max结点希望这个值大。	√

概念题：Max-value函数中什么原因会引起剪枝
A：有一个儿子结点的返回值比当前的beta值大		√
B：有一个儿子结点的返回值比当前的beta值小
C：有一个儿子结点的返回值比当前的alpha值大
D：有一个儿子结点的返回值比当前的alpha值小

概念题：alpha的更新是在什么时候进行的在
A：在得到Max结点的值时
B：在发现Max结点有一个值更大的儿子结点时	√
C：在得到Min结点的值时
D：在Min结点的儿子结点值返回时

概念题：不完美的实时决策主要使用哪些策略来提高效率
A：深度限制
B：终止状态的判断替换成判断是否达到深度
C：非终止状态的效用值用定义的评估函数来计算



· 下面对 Alpha-Beta 剪枝搜索算法描述中正确的是：
节点先后次序对剪枝效率有影响
剪枝对算法输出结果没有影响
大多情况下剪枝会提高算法效率

· 下面对 minimax 搜索算法描述中正确的是：
需要遍历游戏树中所有节点
MAX 节点希望自己收益最大化
给定一个游戏搜索树，minimax 算法通过每个节点的 minimax 值来决定最优策略
MIN 节点希望对方收益最小化

Alpha 和 Beta 两个值在 Alpha-Beta 剪枝搜索中被用来判断某个节点的后续节点是否可 被剪枝，下面对 Alpha 和 Beta 的初始化取值描述正确的是
Alpha 和 Beta 初始值分别为负无穷大和正无穷大

· 下面对上限置信区间 (UCB)算法在多臂赌博机中的描述，哪句描述是不正确的
UCB 算法在探索-利用之间寻找平衡
UCB 算法既考虑拉动在过去时间内获得最大平均奖赏的赌博机，又希望去选择那些拉动臂膀次数最少的赌博机
UCB 算法每次随机选择一个赌博机来拉动其臂膀（X）
UCB 算法是优化序列决策问题的方法

· 下面对 minimax 搜索、alpha-beta 剪枝搜索和蒙特卡洛树搜索的描述中正确的是
minimax 是穷举式搜索
对于一个规模较小的游戏树，alpha-beta 剪枝搜索和 minimax 搜索的结果相同
三种搜索算法中，只有蒙特卡洛树搜索是采样搜索
alpha-beta 剪枝搜索和蒙特卡洛树搜索都是非穷举式搜索

---

概念题：从如下用于约束满足问题 (CSP)的状态表示中选择正确的答案。
A：Factored 因子

应用题：‍设{A, B, C, D}为变量，每个变量的域是{u, v, w}，且“!=”表示不等于，从如下表达式中选择那个是CSP形式化的2元约束？
A：Diff(A, D)

概念题：比较CSP和状态空间搜索，并从下列叙述中选择正确的答案。
A：CSP求解系统会比状态空间搜索求解系统快。
		CSP可以快速排除大的搜索空间样本。

概念题：‍如下陈述中哪些是约束传播局部一致性的正确类型？
A：弧一致、节点一致、k一致、路径一致

概念题：‍从如下有关“回溯搜索”概念中选择正确的答案。
A：每次为变量选择值并且当变量没有合法赋值时回溯。
		递增地构建解的候选，并且一旦确定部分候选c不能成为合法的解，就将c抛弃。

概念题：‏约束满足问题 (CSP) 被定义为其状态必须满足若干_______________的一组对象。
A：约束和限制

概念题：‎在对一个变量选择一个新值时，最小冲突启发式选择导致与其它变量呈现____的值。
A：最少冲突

概念题：‍为了简化约束图为树结构，有两种方法可以采用，即___________和树分解。
A：割集调节

概念选择题：约束满足问题中的目标形式化是：
A：给出具体的目标状态
B：给出目标的要求	√
C：给除目标状态的集合
D：给出符合目标状态要求的装他提变量约束集合

应用题：用标准搜索方法来解决约束满足问题时，假设描述状态的变量总个数为n，变量的取值个数最多为d，在生成的搜索树的第L层有____（多少)个结点，最坏情况下搜索树生产____*d的n次方（多少)个叶子结点。
A：(n-L)d、n!

概念题：回溯搜索算法的本质是深度优先搜索。

概念题：回溯搜索算法中哪些关键步骤能使用策略提高搜索效率
A：如何选择下一个未赋值变量优先进行赋值尝试，即select-unassigned-variable	√
B：增加一个深度限制
C：如何选择赋值顺序，即order-domain-values	√
D：减少分支数

应用题：用回溯搜索算法解决约束满足问题时，如果没有赋值的变量分别有x1,X2,X3,它们的剩余合法赋值分别有3个，2个，2个，而且受x1,X2,X3约束的未赋值变量分别有1个，2个，1个，则此时算法应该选择哪个变量进行赋值尝试？____

应用题：在回溯搜索算法中，在确定好尝试赋值的变量后，如果给这个变量赋值为a会使得其他没有赋值变量的剩余合法赋值个数减少量为3，赋值为b时减少量为2，赋值为c减少量为4，则最先应该给此变量尝试赋什么值？____
A：X2

概念题：前向检查表是约束传播的一种方法，它存储的是
A：各变量的剩余合法赋值

概念题：用局部搜索算法来解决约束满足问题时，关于状态的描述错误的是：
A：状态的评估函数与违反的约束个数有关	√
B：每一个状态都给所有变量赋了值	√
C：每一个状态都是符合所有约束的完整赋值	×



---