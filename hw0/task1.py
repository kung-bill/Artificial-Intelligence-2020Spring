class Node(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution(object):
    res = 0
    def traversal(self, curr):
        if curr != None:
            l_node = curr.left
            r_node = curr.right  
            if l_node != None and l_node.left == None and l_node.right == None:
                self.res += l_node.val
            self.traversal(l_node)
            self.traversal(r_node)
            
        
    def sum_of_left_leaves(self, _root):
        """
        :type _root: Node
        :return type: int
        """
        
        self.traversal(_root)
        return self.res
        
        


if __name__ == '__main__':
    # testing data 1
    root = Node(3)
    root.left = Node(9)
    root.right = Node(20)
    root.right.left = Node(15)
    root.right.right = Node(7)
    sol = Solution()
    print(sol.sum_of_left_leaves(root)) #24
    
    # testing data 2
    root = Node(9)
    root.left = Node(8)
    root.right = Node(6)
    root.left.left = Node(5)
    root.left.right = Node(2)
    root.right.left = Node(1)
    sol = Solution()
    print(sol.sum_of_left_leaves(root)) #6
    
    # testing data 3
    root = Node(20) 
    root.left = Node(9) 
    root.right = Node(49) 
    root.right.left = Node(23)         
    root.right.right = Node(52) 
    root.right.right.left = Node(50) 
    root.left.left = Node(5) 
    root.left.right = Node(12) 
    root.left.right.right = Node(12)
    sol = Solution()
    print(sol.sum_of_left_leaves(root)) #78