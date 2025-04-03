import sys

import pygame
import pymunk
import math
import pymunk.pygame_util
import os
# 资源文件目录访问
def source_path(relative_path):
    # 是否Bundle Resource
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


# 修改当前工作目录，使得资源文件可以被正确访问
cd = source_path('')
os.chdir(cd)
pygame.init()

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 678
BOTTOM_PANEL = 50

# 游戏窗口
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT + BOTTOM_PANEL))
pygame.display.set_caption("台球游戏")
logo = pygame.image.load("assets/favicon.ico")
pygame.display.set_icon(logo)

# pymunk 空间
space = pymunk.Space()
static_body = space.static_body
# space.gravity = (0, 2000)
draw_options = pymunk.pygame_util.DrawOptions(screen)

# 时钟
clock = pygame.time.Clock()
FPS = 120

# 游戏变量
lives = 3
dia = 36
pocket_dia = 66
taking_shot = True
force = 0
max_force = 10000
force_direction = 1
game_running = True
cue_ball_potted = False
powering_up = False
potted_balls = []

# 定义颜色
BG = (50, 50, 50)
RED = (255, 0, 0)
WHITE = (255, 255, 255)

# 字体
font = pygame.font.SysFont("华文楷体", 30)
large_font = pygame.font.SysFont("华文楷体", 60)

# 加载图片
cue_image = pygame.image.load("assets/images/cue.png").convert_alpha()
table_image = pygame.image.load("assets/images/table.png").convert_alpha()
ball_images = []
for i in range(1, 17):
    ball_image = pygame.image.load(f"assets/images/ball_{i}.png").convert_alpha()
    ball_images.append(ball_image)

# 定义绘制文本函数
def draw_text(text, font, text_color, x, y):
    img = font.render(text, True, text_color)
    screen.blit(img, (x, y))

# 为所有的球创建函数
def create_ball(radius, pos):
    body = pymunk.Body()
    body.position = pos
    shape = pymunk.Circle(body, radius)
    shape.mass = 5 # 质量
    shape.elasticity = 0.8 # 添加弹性
    # 使用支点增加摩擦力
    pivot = pymunk.PivotJoint(static_body, body, (0, 0), (0, 0))
    pivot.max_bias = 0 # 偏移为0
    pivot.max_force = 1000 # 线性摩擦力

    space.add(body, shape, pivot)
    return shape

# 创建游戏球体
balls = []
rows = 5
# 遍历创建球
for col in range(5):
    for row in range(rows):
        pos = (250 + (col * (dia + 1)), 267 + (row * (dia + 1)) + (col * dia / 2))
        new_ball = create_ball(dia / 2, pos)
        balls.append(new_ball)
    rows -= 1

# 母球
pos = (888, SCREEN_HEIGHT / 2)
cue_ball = create_ball(dia / 2, pos)
balls.append(cue_ball)

# 在台球桌创建六个球袋
pockets = [
    (55, 63),
    (592, 48),
    (1134, 64),
    (55, 616),
    (592, 629),
    (1134, 616)
]

# 创建台球桌缓冲
cushions = [
    [(88, 56), (109, 77), (555, 77), (564, 56)],
    [(621, 56), (630, 77), (1081, 77), (1102, 56)],
    [(89, 621), (110, 600),(556, 600), (564, 621)],
    [(622, 621), (630, 600), (1081, 600), (1102, 621)],
    [(56, 96), (77, 117), (77, 560), (56, 581)],
    [(1143, 96), (1122, 117), (1122, 560), (1143, 581)]
]

# 创建一个缓冲函数
def create_cushion(poly_dims):
    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    body.position = ((0, 0))
    shape = pymunk.Poly(body, poly_dims)
    shape.elasticity = 0.8 # 给垫子增加弹性
    space.add(body, shape)

# 创建缓冲垫
for c in cushions:
    create_cushion(c)

# 创建球杆
class Cue():
    def __init__(self, pos):
        self.original_image = cue_image
        self.angle = 0
        self.image = pygame.transform.rotate(self.original_image, self.angle)
        self.rect = self.image.get_rect()
        self.rect.center = pos

    def update(self, angle):
        self.angle = angle

    def draw(self, surface):
        self.image = pygame.transform.rotate(self.original_image, self.angle)
        # surface.blit(self.image, self.rect)
        surface.blit(self.image,
                     (self.rect.centerx - self.image.get_width() / 2,
                     self.rect.centery - self.image.get_height() / 2)
                     )

cue = Cue(balls[-1].body.position)

# 创建蓄力显示按钮
power_bar = pygame.Surface((10, 70))
power_bar.fill(RED)
# 游戏循环
run = True
while run:

    clock.tick(FPS)
    space.step(1 / FPS) # 跟踪球下落的位置

    # 填充背景色
    screen.fill(BG)

    # 画台球桌
    screen.blit(table_image, (0, 0))

    # 判断是否有进球
    for i, ball in enumerate(balls):
        for pocket in pockets:
            ball_x_dist = abs(ball.body.position[0] - pocket[0])
            ball_y_dist = abs(ball.body.position[1] - pocket[1])
            ball_dist = math.sqrt((ball_x_dist ** 2) + (ball_y_dist ** 2))
            if ball_dist <= pocket_dia / 2:
                # 判断母球是否进入球袋
                if i == len(balls) - 1:
                    lives -= 1
                    cue_ball_potted = True
                    ball.body.position = (-100, -100)
                    ball.body.velocity = (0.0, 0.0)
                else:
                    space.remove(ball.body)
                    balls.remove(ball)
                    potted_balls.append(ball_images[i])
                    ball_images.pop(i)


    # 画台球
    for i, ball in enumerate(balls):
        screen.blit(ball_images[i], (ball.body.position[0] - ball.radius, ball.body.position[1] - ball.radius))
    # 判断所有的球是否正在运动
    taking_shot = True
    for ball in balls:
        if int(ball.body.velocity[0]) != 0 or int(ball.body.velocity[1]) != 0:
            taking_shot = False


    # 画台球杆
    if taking_shot == True and game_running == True:
        if cue_ball_potted == True:
            # 重新定位母球的位置
            balls[-1].body.position = (888, SCREEN_HEIGHT / 2)
            cue_ball_potted = False
        # 计算台球杆的角度
        mouse_pos = pygame.mouse.get_pos()
        # 重新定位
        cue.rect.center = balls[-1].body.position
        x_dist = balls[-1].body.position[0] - mouse_pos[0]
        y_dist = -(balls[-1].body.position[1] - mouse_pos[1])
        cue_angle = math.degrees(math.atan2(y_dist, x_dist))
        cue.update(cue_angle)
        cue.draw(screen)

    # 台球杆积蓄力量
    if powering_up == True and game_running == True:
        force += 100 * force_direction
        if force >= max_force or force <= 0:
            force_direction *= -1
        # 画力量条
        for b in range(math.ceil(force / 1000)):
            screen.blit(power_bar,
                (balls[-1].body.position[0] - 30 + (b * 15),
                balls[-1].body.position[1] + 30))
    elif powering_up == False and taking_shot == True:
        x_impulse = math.cos(math.radians(cue_angle))
        y_impulse = math.sin(math.radians(cue_angle))
        balls[-1].body.apply_impulse_at_local_point((force * -x_impulse, force * y_impulse), (0, 0))
        force = 0
        force_direction = 1

    # 画底部的面板
    pygame.draw.rect(screen, BG, (0, SCREEN_HEIGHT, SCREEN_WIDTH, BOTTOM_PANEL))
    draw_text("生命值: " + str(lives), font, WHITE, SCREEN_WIDTH - 200, SCREEN_HEIGHT + 10)

    # 画已进入袋子的球
    for i, ball in enumerate(potted_balls):
        screen.blit(ball, (10 + (i * 50), SCREEN_HEIGHT + 10))

    # 检查游戏是否结束
    if lives <= 0:
        draw_text("游戏结束", large_font, WHITE, SCREEN_WIDTH / 2 - 160, SCREEN_HEIGHT / 2 - 180)
        game_running = False

    # 检查所有的球是否都进入球袋
    if len(balls) == 1:
        draw_text("你胜利了!", large_font, WHITE, SCREEN_WIDTH / 2 - 160, SCREEN_HEIGHT / 2 - 180)
        game_running = False

    # 事件处理器
    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN and taking_shot == True:
            powering_up = True
        if event.type == pygame.MOUSEBUTTONUP and taking_shot == True:
            powering_up = False

        if event.type == pygame.QUIT:
            run = False
            pygame.quit()
            sys.exit()
    # space.debug_draw(draw_options)
    pygame.display.update()



