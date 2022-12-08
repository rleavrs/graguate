DS代码

[TOC]

###### 前序递归

```cpp
/*
递归
*/
class Solution {
    public void dfs(List<Integer> ans, TreeNode now) {
        if (now == null) return;
        ans.add(now.val);
        dfs(ans, now.left);
        dfs(ans, now.right);
    }

    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> ans = new ArrayList<>();
        dfs(ans, root);
        return ans;
    }
}

```

###### 前序非递归

```cpp
/*
非递归
*/
class Solution {
    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> ans = new ArrayList<>();
        if (root == null) return ans;

        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        while (!stack.empty()) {
            TreeNode now = stack.pop();
            ans.add(now.val);
            if (now.right != null) {
                stack.push(now.right);
            }
            if (now.left != null) {
                stack.push(now.left);
            }
        }

        return ans;
    }
}

```

###### 前序无栈

```cpp
void preorderMorrisTraversal(TreeNode *root) {
     TreeNode *cur = root, *prev = NULL;
     while (cur != NULL)
     {
         if (cur->left == NULL)
         {
             printf("%d ", cur->val);
             cur = cur->right;
         }
         else
         {
             prev = cur->left;
             while (prev->right != NULL && prev->right != cur)
                 prev = prev->right;

             if (prev->right == NULL)
             {
                 printf("%d ", cur->val);  // the only difference with inorder-traversal
                 prev->right = cur;
                 cur = cur->left;
             }
             else
             {
                 prev->right = NULL;
                 cur = cur->right;
             }
         }
     }
 }
```



###### 中序递归

```cpp
/*
递归
*/
class Solution {
    public void dfs(List<Integer> ans, TreeNode now) {
        if (now == null) return;
        dfs(ans, now.left);
        ans.add(now.val);
        dfs(ans, now.right);
    }

    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> ans = new ArrayList<>();
        dfs(ans, root);
        return ans;
    }
}

```

###### 中序非递归

```cpp
/*
非递归
*/
class Solution {
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> ans = new ArrayList<>();
        if (root == null) return ans;

        Stack<TreeNode> stack = new Stack<>();
        TreeNode p = root;
        while (!stack.empty() || p != null) {
            while (p != null) {
                stack.push(p);
                p = p.left;
            }
            TreeNode now = stack.pop();
            ans.add(now.val);
            if (now.right != null) {
                p = now.right;// p表示右子树，如果为null则表示遍历完了
            }
        }

        return ans;
    }
}

```

###### 中序无栈

```cpp
void inorderMorrisTraversal(TreeNode *root) {
     TreeNode *cur = root, *prev = NULL;
     while (cur != NULL)
     {
         if (cur->left == NULL)          // 1.
         {
             printf("%d ", cur->val);
             cur = cur->right;
         }
         else
         {
             // find predecessor
             prev = cur->left;
             while (prev->right != NULL && prev->right != cur)
                 prev = prev->right;

             if (prev->right == NULL)   // 2.a)
             {
                 prev->right = cur;
                 cur = cur->left;
             }
             else                       // 2.b)
             {
                 prev->right = NULL;
                 printf("%d ", cur->val);
                 cur = cur->right;
             }
         }
     }
 }
```



###### 后序递归

```cpp
/*
递归
*/
class Solution {
    public void dfs(List<Integer> ans, TreeNode now) {
        if (now == null) return;
        dfs(ans, now.left);
        dfs(ans, now.right);
        ans.add(now.val);
    }

    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> ans = new ArrayList<>();
        dfs(ans, root);
        return ans;
    }
}

```

###### 后序非递归

```cpp
/*
非递归
*/
class Solution {
    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> ans = new ArrayList<>();
        if (root == null) return ans;

        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        while (!stack.empty()) {
            TreeNode now = stack.pop();
            ans.add(now.val);
            if (now.left != null) {
                stack.push(now.left);
            }
            if (now.right != null) {
                stack.push(now.right);
            }
        }
        Collections.reverse(ans);

        return ans;
    }
}

```

###### 后序无栈

```cpp
 void reverse(TreeNode *from, TreeNode *to) // reverse the tree nodes 'from' -> 'to'.
 {
     if (from == to)
         return;
     TreeNode *x = from, *y = from->right, *z;
     while (true)
     {
         z = y->right;
         y->right = x;
         x = y;
         y = z;
         if (x == to)
             break;
     }
 }

 void printReverse(TreeNode* from, TreeNode *to) // print the reversed tree nodes 'from' -> 'to'.
 {
     reverse(from, to);

     TreeNode *p = to;
     while (true)
     {
         printf("%d ", p->val);
         if (p == from)
             break;
         p = p->right;
     }

     reverse(to, from);
 }

 void postorderMorrisTraversal(TreeNode *root) {
     TreeNode dump(0);
     dump.left = root;
     TreeNode *cur = &dump, *prev = NULL;
     while (cur)
     {
         if (cur->left == NULL)
         {
             cur = cur->right;
         }
         else
         {
             prev = cur->left;
             while (prev->right != NULL && prev->right != cur)
                 prev = prev->right;

             if (prev->right == NULL)
             {
                 prev->right = cur;
                 cur = cur->left;
             }
             else
             {
                 printReverse(cur->left, prev);  // call print
                 prev->right = NULL;
                 cur = cur->right;
             }
         }
     }
 }
```



###### 层序递归

```cpp
//递归实现层序遍历
//由于层序遍历是一种广度优先的遍历，所以我们需要通过层数控制整个函数的运行
//每轮循环传入要输出的层数
//然后当层数不等于0时，向下递归，每次递归层数减一
//当层数等于0的时候，则说明已经到达了要输出的那一层，递归的结束条件就满足了，直接输出值即可
	void leavlOrder()
	{
		cout << "递归实现层序遍历:";
		int l = leavl(_root);
		for(int i = 0; i < l; ++i)
		{
			leavlOrder(_root, i);
		}
		cout << endl;
	}

	void leavlOrder(BSTNode *node, int l)
	{
		if (node != nullptr)
		{
			if(l==0)
			{ 
				cout << node->_data << " ";
				return;
			}
			leavlOrder(node->_left, l - 1);
			leavlOrder(node->_right, l - 1);
		}
	}

```



###### 层序非递归

```cpp
//非递归层序遍历需要借助一个队列进行操作
//原因是层序遍历是一个广度优先的遍历操作，需要从左至右，从上至下每一层逐个打印
//首先把根结点入队
//然后进行一个大循环，判断队列是否为空
//只要队列不为空，则说明上一层的元素还没有打印完
//将队头出队，打印队头元素，将队头的左孩子和右孩子依次从队尾入队
//直到队内空，则说明遍历完成。
void NonLeavlshow()
	{
		cout << "非递归实现层序遍历:";
		if (_root == nullptr)
		{
			return;
		}
		queue<BSTNode*>myqueue;
		myqueue.push(_root);
		while (!myqueue.empty())
		{
			if (myqueue.front()->_left)
			{
				myqueue.push(myqueue.front()->_left);
			}
			if (myqueue.front()->_right)
			{
				myqueue.push(myqueue.front()->_right);
			}
			cout << myqueue.front()->_data << " ";
			myqueue.pop();
		}
		cout << endl;
	}

```



