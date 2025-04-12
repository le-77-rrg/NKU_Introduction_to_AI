import math
import numpy as np
import random
import copy
import time

class MCTSNode:
    def __init__(self,board,color,parent=None,action=None):
        self.board=board
        self.color=color
        self.parent=parent
        self.children=[]
        self.visits=0
        self.wins=0
        self.action=action
        self.epsilon=0.3
        self.gamma=0.999
        self.untried_actions=list(board.get_legal_actions(color))

    def add_child(self,action):
        temp_board=copy.deepcopy(self.board)
        temp_board._move(action,self.color)
        child_color='O'if self.color=='X'else'X'
        child=MCTSNode(temp_board,child_color,self,action)
        self.children.append(child)
        self.untried_actions.remove(action)
        return child


    def best_child(self,weight_c=1.414):
        # 防止除零错误
        ucb1=[]
        for child in self.children:
            if child.visits == 0:
                ucb1.append(float('inf'))  # 未访问的节点给予最高优先级
            else:
                # 使用更优的UCB参数
                exploitation = child.wins / child.visits
                exploration = weight_c * math.sqrt(2*math.log(self.visits) / child.visits)
                ucb1.append(exploitation + exploration)
        return self.children[np.argmax(ucb1)]

    def get_best_child(self,weight_c=1.414):
        # 防止除零错误
        ucb1=[]
        for child in self.children:
            if child.visits == 0:
                ucb1.append(float('inf'))  # 未访问的节点给予最高优先级
            else:
                # 使用更优的UCB参数 
                ucb1.append(child.wins/child.visits)
        return self.children[np.argmax(ucb1)]        

    def update(self,result,diff):
        self.visits+=1
        new_value=10+diff
        if result==1:
            self.wins+=new_value if self.color=='X'else -(new_value)
        elif result==0:
            self.wins+=new_value if self.color=='O'else -(new_value)
        else:
            pass #平局


    def selection(self,weight):
        cur_Node=self
        while cur_Node.children and not cur_Node.untried_actions:
            if random.random() > self.epsilon:
                cur_Node = cur_Node.best_child()
            else:
                cur_Node = random.choice(cur_Node.children)
            self.epsilon *= self.gamma
        return cur_Node
    
    def expansion(self):
        if self.untried_actions:
            action=random.choice(self.untried_actions)
            return self.add_child(action)
        return self
    

    def simulation(self):
        flag=0
        temp_board=copy.deepcopy(self.board)#在复制的棋盘上进行模拟
        cur_color=self.color
        while True:
            if temp_board.count('X')+temp_board.count('O')== 64 or flag==2:
                winner,diff=temp_board.get_winner()
                return winner,diff #返回模拟胜者
            legal_actions=list(temp_board.get_legal_actions(cur_color))
            if not legal_actions:
                flag+=1
                cur_color='O'if cur_color=='X'else'X'
            else:   
                action=random.choice(legal_actions)
                temp_board._move(action,cur_color)
                cur_color='O'if cur_color=='X'else'X'

    def backpropagation(self,result,diff):
        cur=self
        while cur:
            cur.update(result,diff)
            cur=cur.parent
    
    def MCTS_search(self,iterations=1000000,time_limit=60):
        start=time.time()
        cnt=0
        empty_count = sum(1 for i in range(8) for j in range(8) if self.board._board[i][j] == '.')
        
        # 根据游戏阶段调整探索/利用平衡 
        if empty_count > 40:  # 游戏早期，多探索
            exploration_weight = 2.5  # 大幅增加探索权重
        elif empty_count > 20:  # 游戏中期，平衡探索和利用
            exploration_weight = 1.5  # 增加探索
        else:  # 游戏后期，更注重利用
            exploration_weight = 0.3  # 更注重利用
        
        while cnt<iterations:
            if time.time()-start>time_limit:
                break #超时
            
            node=self.selection(exploration_weight)

            if node.untried_actions:
                node=node.expansion()
            
            result,diff=node.simulation()

            node.backpropagation(result,diff)
            cnt+=1

        return self.get_best_child(weight_c=exploration_weight).action



class AIPlayer:
    """
    AI 玩家
    """

    def __init__(self, color):
        """
        玩家初始化
        :param color: 下棋方，'X' - 黑棋，'O' - 白棋
        """

        self.color = color

    def get_move(self, board):
        """
        根据当前棋盘状态获取最佳落子位置
        :param board: 棋盘
        :return: action 最佳落子位置, e.g. 'A1'
        """
        if self.color == 'X':
            player_name = '黑棋'
        else:
            player_name = '白棋'
        print("请等一会，对方 {}-{} 正在思考中...".format(player_name, self.color))

        # -----------------请实现你的算法代码--------------------------------------
        action = MCTSNode(board, self.color).MCTS_search(time_limit=30)
        # ------------------------------------------------------------------------

        return action