import pygame
import pymunk
import pymunk.pygame_util
import math
import numpy as np
from collections import defaultdict
import csv
#from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# 添加用于记录数据的列表
shot_records = []
# common
TITLE = "Pool Game"  # 游戏标题
SCREEN_WIDTH = 1200  # 窗口宽度
SCREEN_HEIGHT = 678   # 窗口高度
# 底部面板高度
BOTTOM_PANEL = 50
BACKGROUND_COLOR = (50, 50, 50)  # 背景颜色
TEXT_COLOR = (255, 255, 255)  # 文字颜色

# ball data
MAX_BALL = 17  # 最大球数
BALL_MASS = 5  # 球的质量
BALL_ELASTICITY = 0.9  # 球的弹性
BALL_DIAMETER = 36  # 球的直径

# wall data
FRICTION = 980  # 摩擦力
CUSHION_ELASTICITY = 0.6   # 靠垫的弹性
POCKET_DIAMETER = 70  # 球袋的直径

# shooting data
MAX_FORCE = 20000  # 最大击球力度
FORCE_STEP = 1000  # 力度变化步长
FORCE_OPTIONS = [2000, 3500, 4500, 5500, 6500, 11500, 8500, 9500]  # 可选的击球力度
#FORCE_OPTIONS = [1000, 1500, 1500, 4500, 1500, 1500, 1500, 1500]
player_targets = {1: None, 2: None}   # 玩家目标初始化
target_group_printed = False  # 目标组是否已打印
target_group = None  # 'solids', 'stripes', 或 None
first_potted = False  # 是否是第一次进袋
shot_data = []  # 记录击球数据

# power bar
BAR_WIDTH = 10  # 力度条宽度
BAR_HEIGHT = 20  # 力度条高度
BAR_SENSTIVITY = 1000  # 力度条灵敏度
BAR_COLOR = (255, 0, 0)  # 力度条颜色

# create six pockets on table
POCKETS = [(55, 63), (592, 48), (1134, 64), (55, 616), (592, 629), (1134, 616)]
ball_positions_history = {i: [] for i in range(1, MAX_BALL)}  # 初始化每个球的位置历史记录

# create pool table cushions 创建台球桌的靠垫
CUSHIONS = [
    [(88, 56), (109, 77), (555, 77), (564, 56)],
    [(621, 56), (630, 77), (1081, 77), (1102, 56)],
    [(89, 621), (110, 600), (556, 600), (564, 621)],
    [(622, 621), (630, 600), (1081, 600), (1102, 621)],
    [(56, 96), (77, 117), (77, 560), (56, 581)],
    [(1143, 96), (1122, 117), (1122, 560), (1143, 581)],
]

# initilize the modules
pygame.init()  # 初始化pygame模块

# 字体设置
font = pygame.font.SysFont("Lato", 30)
large_font = pygame.font.SysFont("Lato", 60)

# clock
FPS = 120  # 帧率设置
clock = pygame.time.Clock()  # 创建时钟对象

# game window
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT + BOTTOM_PANEL))
pygame.display.set_caption(TITLE)

# pymunk space
space = pymunk.Space()  # 创建pymunk空间
static_body = space.static_body  # 静态物体
draw_options = pymunk.pygame_util.DrawOptions(screen)  # 设置绘图选项

# game variables
lives = 5  # 剩余生命值
force = 0  # 当前力度
force_direction = 1  # 力度方向
game_running = True  # 游戏是否在运行
cue_ball_potted = False  # 白球是否进袋
taking_shot = True  # 是否在击球
powering_up = False  # 是否在蓄力
potted_balls = []  # 已进袋的球

# load images
cue_image = pygame.image.load("assets/images/cue.png").convert_alpha()  # 加载球杆图像
table_image = pygame.image.load("assets/images/table.png").convert_alpha()  # 加载桌子图像
ball_images = []  # 创建球的图像列表
for i in range(1, MAX_BALL):  # 遍历球的编号
    ball_image = pygame.image.load(f"assets/images/ball_{i}.png").convert_alpha()  # 加载每个球的图像
    ball_images.append(ball_image)   # 将图像添加到列表


def shoot_cue_ball(force, angle):
    # 计算力的分量
    fx = math.cos(angle) * force  # 计算x方向的力
    fy = math.sin(angle) * force  # 计算y方向的力
    # 将力施加到白球上
    cue_ball.body.apply_impulse_at_local_point((fx, fy), (0, 0))  # 在白球上施加冲击力

    # 记录击球力度和初始速度
    initial_velocity = cue_ball.body.velocity.length   # 获取白球的初始速度
    shot_data.append((force, initial_velocity))  # 将击球力度和初始速度记录到数据中

def reset_ball_positions():
    for ball in balls:  # 遍历所有球
        if ball_positions_history[ball.number]:  # 如果有历史位置记录
                last_position = ball_positions_history[ball.number][-1]  # 获取最后记录的位置
                ball.body.position = last_position  # 重置球的位置


def reset_game():
    global balls, space, game_running, cue_ball_potted, lives, potted_balls   # 声明全局变量
    # 清除所有球体和约束
    for ball in balls:  # 遍历所有球
        space.remove(ball.body, ball.shape)  # 从空间中移除球体和形状

    # 重新创建球和其他游戏元素
    balls = setup_balls()  # 假设这是一个设置初始球位置的函数
    cue_ball_potted = False  # 白球进袋状态重置
    game_running = True  # 游戏状态设置为运行
    lives = 5  # 重置生命值
    potted_balls = []  # 重置已进袋球的列表
# function for outputting text onto the screen
def draw_text(text, font, text_col, x, y):
    screen.blit(font.render(text, True, text_col), (x, y)) # 在指定位置绘制文本


# function for creating balls
def create_ball(number,radius, pos):
    body = pymunk.Body()  # 创建球的物理体
    body.position = pos  # 设置球的位置
# 对每个球初始化一个空列表
    shape = pymunk.Circle(body, radius)  # 创建球的形状
    shape.mass = BALL_MASS  # 设置球的质量
    shape.elasticity = BALL_ELASTICITY  # 设置球的弹性
    shape.number = number  # 给球添加编号
    shape.is_potted = False  # 初始化进袋状态为False
    if number == 8:
        shape.type = 'black'  # 设置类型为黑色
    elif 1 <= number <= 7:
        shape.type = 'solids'  # 全色
    elif 9 <= number <= 15:
        shape.type = 'stripes'  # 花色
    # use pivot joint to add friction 使用铰链连接以增加摩擦
    pivot = pymunk.PivotJoint(static_body, body, (0, 0), (0, 0)) # 创建铰链连接
    pivot.max_bias = 0  # disable joint correction # 禁用关节校正
    pivot.max_force = FRICTION  # emulate linear friction # 模拟线性摩擦
    space.add(body, shape, pivot)  # 将球体、形状和铰链添加到空间
    return shape


# setup game balls
balls = []  # 创建球的列表
potted_balls_status = {i: False for i in range(1, 17)}  # 初始化每个球的进袋状态为False

def ball_potted(ball_number):
    """检查指定编号的球是否已经被打进"""
    return potted_balls_status[ball_number]

def all_balls_potted(group):
    """检查一组球是否都已被打进"""
    return all(potted_balls_status[ball_number] for ball_number in group)
def update_potted_status(ball_number):
    """更新指定编号球的进袋状态为True"""
    potted_balls_status[ball_number] = True

rows = 5
ball_number = 1 # 起始球编号
# 假设 BALL_DIAMETER 和 create_ball 已定义
BALL_DIAMETER = 36  # 例如，球的直径为 36

#手动指定每个球的位置
positions = [
    #(1096,590)
    (445,225)
    #(590, 330),
    # (55, 63),
    #  (57, 56),
    #  (109, 77)
    # (57, 63)
    # (58, 63)
    # (59, 63)
    # (51, 63)
    # (52, 63)
    # (53, 63)
    # (54, 63)
    # (55, 61)
    # (55, 62)
    # (343, 360),
    # (374, 391),
    # (405, 422),
    # (436, 453),
    # (467, 484),
    # (498, 515),
    # (529, 546),
    # (650, 367),
    # (750, 467),
    # (850, 567),
    # (550, 267),
    # (250, 367),
]
# cue ball
pos = (267, 452)

balls = []
ball_number = 1  # 起始球编号

# 创建球并将其添加到 balls 列表
for position in positions:
    ball = create_ball(ball_number, BALL_DIAMETER / 2, position)
    balls.append(ball)
    ball_number += 1  # 更新球的编号

#potting balls
# for col in range(5):
#     for row in range(rows):
#         balls.append(
#             create_ball( ball_number,
#                BALL_DIAMETER / 2,
#                 (
#                     250 + col * (BALL_DIAMETER + 1),
#                     267 + row * (BALL_DIAMETER + 1) + col * BALL_DIAMETER / 2,
#                 ),
#             )
#         )
#     #    balls.append(ball)
#         ball_number += 1  # 更新球的编号
#     rows -= 1
# # cue ball
# pos = (888, 544)
# for col in range(5):
#    for row in range(rows):
#        ball = create_ball(
#            ball_number,
#            BALL_DIAMETER / 2,
#            (590 + col * (BALL_DIAMETER + 1), 117 + row * (BALL_DIAMETER + 1) + col * BALL_DIAMETER / 2)
#        )
#        balls.append(ball)
#        ball_number += 1  # 更新球的编号
#    rows -= 1



cue_ball = create_ball(0,BALL_DIAMETER / 2, pos)
balls.append(cue_ball)
# 初始化球的分组和状态
ball_groups = {'stripes': set(), 'solids': set(), 'eight_ball': 8}
target_group = None  # None 表示未决定，可以是 'stripes' 或 'solids'
current_player = 1
break_shot = True

def validate_break_shot(balls):
    """ 验证开球是否有效 """
    potted = [ball for ball in balls if ball.is_potted]
    hits = [ball for ball in balls if ball.hit_cushion]
    if len(potted) >= 1 or len(hits) >= 4:
        return True
    return False

def check_first_potted_ball():
    global player_groups
    updated = False
    for ball_number in range(1, MAX_BALL):  # 假设有一个方法来检查每个球号是否进袋
        if ball_potted(ball_number):  # 假设 `ball_potted` 函数检查球是否进袋
            if ball_number in range(1, 8):
                player_groups[current_player] = 'solids'
                player_groups[3 - current_player] = 'stripes'
                updated = True
                break
            elif ball_number in range(9, 15):
                player_groups[current_player] = 'stripes'
                player_groups[3 - current_player] = 'solids'
                updated = True
                break
    return updated



#def check_first_potted_ball():
#    """ 检查并更新玩家的目标球组 """
#   for ball in balls:
#        if ball.is_potted :  # 只关心已经进袋的球
#            if ball_number in range(1, 8):
#                target_group[1] = 'solids'  # 全色球
#                target_group[2] = 'stripes'  # 花色球
#            elif ball_number in range(9, 16):
#                target_group[1] = 'stripes'
#                target_group[2] = 'solids'
#            break
#        return True  # 只更新一次，找到第一个进袋的球就停止
#    return False
          # 如果没有球进袋或者还未决定球组



# function for creating cushions
def create_cushion(poly_dims):
    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    # body.position = (0, 0)
    shape = pymunk.Poly(body, poly_dims)
    shape.elasticity = CUSHION_ELASTICITY
    space.add(body, shape)


for cushion in CUSHIONS:
    create_cushion(cushion)


# create pool cue
class Cue:
    def __init__(self, pos):
        self.original_image = cue_image
        self.angle = 0
        self.image = pygame.transform.rotate(self.original_image, self.angle)
        self.rect = self.image.get_rect()
        self.rect.center = pos

    def update(self, angle):
        self.angle = math.degrees(angle)

    def draw(self, surface):
        self.image = pygame.transform.rotate(self.original_image, self.angle)
        surface.blit(
            self.image,
            (
                self.rect.centerx - self.image.get_width() / 2,
                self.rect.centery - self.image.get_height() / 2,
            ),
        )


cue = Cue(balls[-1].body.position)

# create power bars to show how hard the cue ball will be hit
power_bar = pygame.Surface((BAR_WIDTH, BAR_HEIGHT))
power_bar.fill(BAR_COLOR)


# MCTS AI CODE

def angle_between_points(point1, point2):
        return math.atan2(point2[1] - point1[1], point2[0] - point1[0])

def angle_difference(angle1, angle2):
    return abs((angle1 - angle2 + math.pi) % (2 * math.pi) - math.pi)

def can_reach_target(current_ball: object, target_ball, other_balls):
    """

    :type current_ball: object
    """
    a = current_ball.body.position[1] - target_ball.body.position[1]
    b = target_ball.body.position[0] - current_ball.body.position[0]
    c = (current_ball.body.position[0] - target_ball.body.position[0]) * current_ball.body.position[1] + (target_ball.body.position[1] - current_ball.body.position[1]) * current_ball.body.position[0]
    
    for ball in other_balls:
        if ball == target_ball:
            continue
        
        dist = ((abs(a * ball.body.position[0] + b * ball.body.position[1] + c)) / math.sqrt(a * a + b * b))
        
        if dist <= BALL_DIAMETER / 2:
            return False

    # Check if the target ball can reach at least one pocket
    target_angle = angle_between_points(current_ball.body.position, target_ball.body.position)
    for pocket in POCKETS:
        pocket_angle = angle_between_points(target_ball.body.position, pocket)
        if angle_difference(target_angle, pocket_angle) > math.pi / 2:
            continue

        a = target_ball.body.position[1] - pocket[1]
        b = pocket[0] - target_ball.body.position[0]
        c = (target_ball.body.position[0] - pocket[0]) * target_ball.body.position[1] + (pocket[1] - target_ball.body.position[1]) * target_ball.body.position[0]

        isClear = True
        for ball in other_balls:
            if ball == target_ball:
                continue

            dist = ((abs(a * ball.body.position[0] + b * ball.body.position[1] + c)) / math.sqrt(a * a + b * b))

            if dist <= BALL_DIAMETER / 2:
                isClear = False

        if isClear: return True

    return False

class MonteCarloTreeSearchNode():
    def __init__(self, state, parent=None, parent_action=None):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self._number_of_visits = 0
        self._results = defaultdict(int)
        self._results[1] = 0
        self._results[-1] = 0
        self._untried_actions = None
        self._untried_actions = self.untried_actions()
        return
    
    def untried_actions(self):
        self._untried_actions = self.get_legal_actions()
        return self._untried_actions
    
    def q(self):
        wins = self._results[1]
        loses = self._results[-1]
        return wins - loses

    def n(self):
        return self._number_of_visits

    def expand(self):

        self._untried_actions.pop(0)
        next_state = self.move(0)
        child_node = MonteCarloTreeSearchNode(
            next_state, parent=self, parent_action=0)

        self.children.append(child_node)
        return child_node
    
    def is_terminal_node(self):
        return self.is_game_over()

    def rollout(self):
        current_rollout_state = self.state
        
        while not self.is_game_over(state = current_rollout_state):
            
            possible_moves = self.get_legal_actions(state = current_rollout_state)
            
            # If no balls can be hole'd then consider it a failed attempt
            if len(possible_moves) == 0:
                break
            
            action = self.rollout_policy(possible_moves)
            current_rollout_state = self.move(action, state = current_rollout_state)
            if current_rollout_state is None:
                continue  # Skip if the move leads to an invalid state
        return self.game_result(state = current_rollout_state)
    
    def backpropagate(self, result):
        self._number_of_visits += 1.
        self._results[result] += 1.
        if self.parent:
            self.parent.backpropagate(result)
            if result == -1:  # Penalize the path heavily if it leads to a loss
                self._results[result] *= 3  # Double the penalty
            
    def is_fully_expanded(self):
        return len(self._untried_actions) == 0

    def best_child(self, c_param=1.4):
        choices_weights = [(c.q() / c.n()) + c_param * np.sqrt((2 * np.log(self.n()) / c.n())) for c in self.children]
        return self.children[np.argmax(choices_weights)]
    
    def rollout_policy(self, possible_moves):
        # if np.random.random() < 0.3:  # 30% 的概率随机选择一个行动
        #     return np.random.choice(possible_moves)
            # 假设 possible_moves 是包含球对象的列表
            #closest_index = 0
            #min_distance = float('inf')
            best_index = -1
            best_score = float('-inf')  # 初始化为极小值，寻找最大评分
            cue_ball_position = self.state[-1].body.position  # 假设母球总是列表的最后一个元素
            #for i, ball in enumerate(possible_moves):
            #       distance = cue_ball_position.get_distance(ball.body.position)
            #       if distance < min_distance:
            #         min_distance = distance
            #         closest_index = i
            #return closest_index  # 返回最接近母球的球的索引
            for i, ball in enumerate(possible_moves):
                distance = cue_ball_position.get_distance(ball.body.position)
                path_clear = self.path_is_clear(cue_ball_position, ball.body.position)
                #angle_to_pocket = self.calculate_angle_to_pocket(ball.body.position)
                # 评分函数可以根据实际需要调整
                score = self.evaluate_shot(distance, path_clear)

                if score > best_score:
                    best_score = score
                    best_index = i

            return best_index  # 返回评分最高的球的索引

    def path_is_clear(self, start_pos, end_pos):
        a = end_pos[1] - start_pos[1]
        b = start_pos[0] - end_pos[0]
        c = end_pos[0] * start_pos[1] - start_pos[0] * end_pos[1]

        for ball in self.state[:-1]:  # 假设最后一个球是白球
            ball_pos = ball.body.position
            if ball_pos != start_pos and ball_pos != end_pos:
                distance = abs(a * ball_pos[0] + b * ball_pos[1] + c) / math.sqrt(a ** 2 + b ** 2)
                if distance <= ball.radius:
                    return False

        return True



    def evaluate_shot(self, distance, path_clear):
                # 评分可能考虑距离短、路径清晰和角度适中
                score = 0
                if path_clear:
                    score += 100  # 路径清晰得高分
                score -= distance  # 距离越远分数越低
                #score += abs(math.degrees(angle_to_pocket))  # 角度适中得分
                return score

        ##if np.random.random() < 0.1:  # 10% 的概率随机选择一个行动
        ##    return np.random.choice(possible_moves)
        ##else:
            # 选择最接近的球，先计算距离然后选择最小距离的球
        ##    distances = [ball.body.position.get_distance(self.state[-1].body.position) for ball in possible_moves]
        ##    return possible_moves[np.argmin(distances)]
        # Chose the closest valid ball
        #closest = 99
        #chosen = 0
        #for i, ball in enumerate(possible_moves):
        #    dist = ball.body.position.get_distance(self.state[-1].body.position)
        #    if dist < closest:
        #        closest = dist
        #        chosen = i

        #return chosen

    def _tree_policy(self):
        current_node = self
        while not current_node.is_terminal_node():

            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
                if current_node in [child for child in self.children if self._results[-1] > 3]:
                    continue  # Avoid repeating known bad actions frequently
        return current_node
    
    def best_action(self):
        simulation_no = 200

        for i in range(simulation_no):

            v = self._tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)

        return self.best_child(c_param=0.)

    # Action is an index in the state list for the ball to shoot at
    # Since only the cue_ball can be shot from, we change it's body.position to the position of the chosen action
    
    def get_legal_actions(self, state = None):
        '''
        Modify according to your game or
        needs. Constructs a list of all
        possible actions from current state.
        Returns a list.
        '''
        # This function should calculate and return each ball in state, that could be hole'd by 1 action
        # This will be the bulk of the actual computation
        if state == None:
            state = self.state

        current_ball = state[-1]
        actions = []
        for target in state[:-1]:
            if can_reach_target(current_ball, target, state[:-1]) and target.type == target_group:
                actions.append(target)

        return actions if actions else [state[0]]
        #actions = [target for target in state[:-1] if can_reach_target(current_ball, target, state[:-1])]
        #if len(actions) == 0:
        #    return [state[0]]
        #else:
        #    return actions
    
    def is_game_over(self, state = None):
        '''
        Modify according to your game or 
        needs. It is the game over condition
        and depends on your game. Returns
        true or false
        '''
        # Returns wether or not only the cue_ball is left
        # Lives have not been implemented in the logic (yet)
        if state == None:
            state = self.state
            
        return len(state) <= 1
        
    def game_result(self, state = None):
        '''
        Modify according to your game or 
        needs. Returns 1 or 0 or -1 depending
        on your state corresponding to win,
        tie or a loss.
        '''
        if state == None:
            state = self.state
        if any(ball.number == 8 and ball.is_potted for ball in state):
            # 检查目标球组是否已被清空
            if not all(ball.is_potted for ball in state if ball.number in target_group):
                return -1000  # 重罚
            return 1000  # 如果8号球是最后一个球，返回正奖励
        if len(state) > 1:
            return 0
        if len(state) == 1:
            return 1
        if len(state) < 1:
            return -1



    def move(self,action, state = None):
        '''
        Modify according to your game or 
        needs. Changes the state of your 
        board with a new value. For a normal
        Tic Tac Toe game, it can be a 3 by 3
        array with all the elements of array
        being 0 initially. 0 means the board 
        position is empty. If you place x in
        row 2 column 3, then it would be some 
        thing like board[2][3] = 1, where 1
        represents that x is placed. Returns 
        the new state after making a move.
        '''
        if state == None:
            state = self.state
        # Set the cue_ball position to the selected ball
        # This is wildly inaccurate, and could be improved by calculating approximate landing after rolling
        state[-1] = state[action]

        # Remove the selected ball. We assume it has been hole'd
        state.pop(action)

        return state
    
# END OF AI CODE

# game loop
game_on = True
just_shot = False

while game_on:
    clock.tick(FPS)
    space.step(1 / FPS)

    # fill background
    screen.fill(BACKGROUND_COLOR)


    def draw_end_game():
        draw_text("8-ball potted incorrectly. Restarting...", large_font, TEXT_COLOR, SCREEN_WIDTH / 2 - 200,
                  SCREEN_HEIGHT / 2)
        pygame.display.update()
        pygame.time.wait(2000)  # 等待2秒后重启游戏

    for ball in balls:
        for pocket in POCKETS:
            if (ball.body.position.get_distance(pocket) <= POCKET_DIAMETER / 2):
                ball.is_potted = True
                update_potted_status(ball)  # 可能还需要更新其他相关的状态或统计
                print(f"Ball {ball.number} potted.")  # 输出进袋的球的编号
                if not first_potted:  # 首次进袋决定目标球组
                    first_potted = True
                    if ball.number == 8:
                        print("Game over: 8-ball potted incorrectly.")
                        balls[7].body.position = (888, SCREEN_HEIGHT / 2)
                        ball.is_potted = False
                        first_potted = False
                        #lives= -1
                        #game_running = False
                        #reset_game()
                        #if last_node:
                        #    last_node.backpropagate(-1)  # 假设-1表示不良的结果
                    elif 1 <= ball.number <= 7:
                        target_group = 'solids'
                        print("Target group is solids.")
                    elif 9 <= ball.number <= 15:
                        target_group = 'stripes'
                        print("Target group is stripes.")

    for ball in balls:
        if ball_potted(ball_number):  # 假设这是一个检查球是否进袋的函数
            ball.is_potted = True
            update_potted_status(ball.number)  # 这个函数更新全局进袋状态记录

    # draw pool table
    screen.blit(table_image, (0, 0))

    # check if any balls have been potted
    for i, ball in enumerate(balls):
        for pocket in POCKETS:
            if (
                math.sqrt(
                    (abs(ball.body.position[0] - pocket[0]) ** 2)
                    + (abs(ball.body.position[1] - pocket[1]) ** 2)
                )
                <= POCKET_DIAMETER / 2

            ):
                ball.body.position = (-444, -444)
                ball.body.velocity = (0.0, 0.0)
                # check if the potted ball was the cue ball
                if i == len(balls) - 1:
                    lives -= 1
                    cue_ball_potted = True
                else:
                    space.remove(ball.body)
                    balls.remove(ball)
                    potted_balls.append(ball_images[i])
                    ball_images.pop(i)

    # draw pool balls
    for i, ball in enumerate(balls):
        screen.blit(
            ball_images[i],
            (ball.body.position[0] - ball.radius, ball.body.position[1] - ball.radius),
        )

    taking_shot = True

    # check if all the balls have stopped moving
    for ball in balls:
        if int(ball.body.velocity[0]) != 0 or int(ball.body.velocity[1]) != 0:
            taking_shot = False

    # draw pool cue
    if just_shot:
        pygame.time.wait(1000)
        just_shot = False


    def choose_force(distance):
        # 基于目标位置选择力度，这里的逻辑可以根据实际需要定制
        # 例如，距离越远选择越大的力度
        if distance < 200:
            return FORCE_OPTIONS[1]
        elif distance < 220:
            return FORCE_OPTIONS[2]
        elif distance < 260:
            return FORCE_OPTIONS[3]
        elif distance < 300:
            return FORCE_OPTIONS[4]
        else:
            return FORCE_OPTIONS[5]
        
    if taking_shot and game_running:
        
        if cue_ball_potted:
            # reposition cue ball
            balls[-1].body.position = (888, SCREEN_HEIGHT / 2)
            cue_ball_potted = False
        # calculate pool cue angle
        cue.rect.center = balls[-1].body.position
        # mouse_pos = pygame.mouse.get_pos()
        # cue_angle = math.degrees(
        #     math.atan2(
        #         -(balls[-1].body.position[1] - mouse_pos[1]),
        #         balls[-1].body.position[0] - mouse_pos[0],
        #     )
        # )
        
        # Calculate angle to shoot with AI
        
        root = MonteCarloTreeSearchNode(state = balls[:])
        selected_node = root.best_action()
        
        target = selected_node.state[-1]
        
        # Re-calculate which pocket to aim for
        valid_pockets = []
        distance_to_pocket = float('inf')
        target_angle = angle_between_points(balls[-1].body.position, target.body.position)
        for pocket in POCKETS:
            pocket_angle = angle_between_points(target.body.position, pocket)
            if angle_difference(target_angle, pocket_angle) > math.pi / 2:
                continue

            a = target.body.position[1] - pocket[1]
            b = pocket[0] - target.body.position[0]
            c = (target.body.position[0] - pocket[0]) * target.body.position[1] + (pocket[1] - target.body.position[1]) * target.body.position[0]

            isClear = True
            for ball in balls:
                if ball == target:
                    continue

                
                dist = ((abs(a * ball.body.position[0] + b * ball.body.position[1] + c)) / math.sqrt(a * a + b * b))
                
                if dist <= BALL_DIAMETER / 2:
                    isClear = False
                    
            if isClear: valid_pockets.append(pocket)
        
        closest = 99999
        chosen = 0
        for j, pocket in enumerate(valid_pockets):
            dist = pymunk.Vec2d(pocket[0], pocket[1]).get_distance(target.body.position)
            if dist < closest:
                closest = dist
                chosen = j
        
        target_pos = (target.body.position[0], target.body.position[1])
        if len(valid_pockets) != 0:
            chosen_pocket = valid_pockets[chosen]
            angle = angle_between_points(chosen_pocket, target_pos)
            x = BALL_DIAMETER * math.cos(angle)
            y = BALL_DIAMETER * math.sin(angle)

            target_pos = target_pos[0]+x, target_pos[1]+y

        print("\nBall Positions before shot:")
        for i, ball in enumerate(balls[:-1]):  # 不包括母球
            if not ball.is_potted:  # 检查球是否未进袋
                x, y = ball.body.position
                print(f"Ball {ball.number} Position: ({x:.2f}, {y:.2f})")
                #ball_positions_history[ball.number].append((x, y))
        # 假设某个球进袋了
        #update_potted_status(ball_number_that_potted)


        # 对于母球的位置，单独处理
        x, y = balls[-1].body.position
        print(f"Cue Ball Position: ({x:.2f}, {y:.2f})")
        #ball_positions_history[balls[-1].number].append((x, y))
        print(valid_pockets)
        cue_angle = angle_between_points(balls[-1].body.position, target_pos)

        print(f"\nShooting with angle: {math.degrees(cue_angle)} degrees")

        # 这里是确定距离的部分
        target = selected_node.state[-1]  # 假设这是你的目标球
        target_position = target.body.position  # 假设这是Vec2d类型
        closest_distance = float('inf')
        chosen_pocket = None

        #target_group = check_first_potted_ball()

        if not target_group_printed:
            if check_first_potted_ball():  # 此函数调用后会更新 player_groups
                target_group = player_groups[current_player]
                if target_group is not None:
                    print(f"AI's target balls are now: {target_group}")
                    target_group_printed = True

        #if target_group:
        #    print(f"AI's target balls are now: {target_group}")
            # 只在球组未被决定时调用


        for pocket in POCKETS:
            distance = target.body.position.get_distance(pocket)
            if distance < closest_distance:
                closest_distance = distance

        chosen_force = choose_force(closest_distance)


        cue.update(angle_between_points(balls[-1].body.position, (balls[-1].body.position[0] + -math.cos(cue_angle), balls[-1].body.position[1] + math.sin(cue_angle))))
        cue.draw(screen)

        # Hardcoded to always shoot at max force
        balls[-1].body.apply_impulse_at_local_point(
            (
                chosen_force * math.cos(cue_angle),
                chosen_force * math.sin(cue_angle),
            ),
            (0, 0),
        )
        just_shot = True
        print(f"Shooting with force: {chosen_force}")
        for force, initial_velocity in shot_data:
            print(f"Force: {force}, Initial Velocity: {initial_velocity}")

    # power up pool cue
    # if powering_up and game_running:
    #     force += FORCE_STEP * force_direction
    #     if force >= MAX_FORCE or force <= 0:
    #         force_direction *= -1
    #     # draw power bars
    #     for adjustment in range(math.ceil(force / BAR_SENSTIVITY)):
    #         screen.blit(
    #             power_bar,
    #             (
    #                 balls[-1].body.position[0] - 70 + adjustment * 15,
    #                 balls[-1].body.position[1] + 30,
    #             ),
    #         )
    # elif not powering_up and taking_shot:
    #     balls[-1].body.apply_impulse_at_local_point(
    #         (
    #             force * -math.cos(math.radians(cue_angle)),
    #             force * math.sin(math.radians(cue_angle)),
    #         ),
    #         (0, 0),
    #     )
    #     force = 0
    #     force_direction = 1

    # draw bottom panel
    pygame.draw.rect(
        screen, BACKGROUND_COLOR, (0, SCREEN_HEIGHT, SCREEN_WIDTH, BOTTOM_PANEL)
    )

    # draw potted balls in bottom panel
    for i, ball in enumerate(potted_balls):
        screen.blit(ball, (10 + (i * 50), SCREEN_HEIGHT + 10))

    # draw lives
    draw_text(
        f"LIVES: {str(lives)}", font, TEXT_COLOR, SCREEN_WIDTH - 200, SCREEN_HEIGHT + 10
    )

    # check for game over
    if lives <= 0:
        draw_text(
            "GAME OVER",
            large_font,
            TEXT_COLOR,
            SCREEN_WIDTH / 2 - 160,
            SCREEN_HEIGHT / 2 - 100,
        )
        game_running = False

    elif ball_potted(8) and not all_balls_potted(target_group[current_player]):
        # 如果8号球被提前打进
        draw_text(
            "GAME OVER - 8 BALL POTTED EARLY",
            large_font,
            TEXT_COLOR,
            SCREEN_WIDTH / 2 - 260,
            SCREEN_HEIGHT / 2 - 100,
        )
        game_running = False

    # check if all balls are potted
    if len(balls) == 1:
        draw_text(
            "YOU WIN",
            large_font,
            TEXT_COLOR,
            SCREEN_WIDTH / 2 - 160,
            SCREEN_HEIGHT / 2 - 100,
        )
        game_running = False

    # event handler
    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN and taking_shot:
            powering_up = True
        if event.type == pygame.MOUSEBUTTONUP and taking_shot:
            powering_up = False
        if event.type == pygame.QUIT:
            game_on = False

    # space.debug_draw(draw_options)
    pygame.display.update()

pygame.quit()

