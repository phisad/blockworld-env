import numpy as np
import gym
from gym import spaces
from gym.envs.classic_control.rendering import Line
from gym.utils import seeding

from gym.envs.classic_control import rendering
from pyglet.gl import GL_POLYGON
from pyglet.gl import glBegin, glVertex3f, glEnd, glColor4f
from pyglet.window import MouseCursor
import math
import pyglet
from pyglet.gl import GL_QUADS, GL_LINE_LOOP, glLineWidth

DEBUG = False

WINDOW_WIDTH = 640
WINDOW_HEIGHT = 640

INPUT_WIDTH = 224
INPUT_HEIGTH = 224

DIGIT_SIZE = 24
# This resolves into 13,12 discrete position on the board (if side-by-side)
# This should be fine, because the remaining space on the edges is smaller than half the block length
# (which is 6 pixels on 640 window size) Thus the block will always fall into the last discrete space
BLOCK_LENGTH = .1524

# This is block size of 48 for window size 640
# This is block size of 34 for window size 448
BLOCK_LENGTH_WINDOW = int(BLOCK_LENGTH * (WINDOW_HEIGHT / 2))  # adjust block size to given resolution

COLOR_BACKGROUND = 1., 1., 180 / 255, 1.
COLOR_BLACK = 0., 0., 0., 1.
COLOR_GREY = .8, .8, .8, 1.
COLOR_RED = 1., 0., 0., 1.
COLOR_GREEN = 0., 1., 0., 1.
COLOR_BLUE = 0., 0., 1., 1.
COLOR_ORANGE = 1., 100 / 255, 30 / 255, 1.
COLOR_YELLOW = 1., 1., 0., 1.

pyglet.options["debug_gl"] = False

MOUSE_SIZE = 8
BIN_COUNT = 10

# parameters given in 'params' (might often change on reset)
PARAM_NUM_BLOCKS = "num_blocks"
""" num_blocks: int or list. 
    When an int is given, then a range(1, num_blocks + 1) is used. Thus, when int, the block id's are one-based, 
    but callers might provide any digits as list. A block digit decoration shows the given block id. """
PARAM_WORLD_STATE = "world_state"

# task parameters given in 'task' (rarely change on reset)
TPARAM_BLOCK_PLACEMENT = "block_placement"  # one of [random, world_state]. The initial block placement.
TPARAM_MOUSE_PLACEMENT = "mouse_placement"  # one of [random, center, none]. The initial mouse placement.
TPARAM_BLOCK_DECORATION = "block_decoration"  # one of [none,digit,logo]
TPARAM_BLOCK_COLORS = "block_colors"  # mapping of block id to color (id's are one-based)

# Mouse handling
STATE_NONE = 0
STATE_MOVED = 1
STATE_PRESSED = 2
STATE_RELEASED = 3
STATE_DRAGGED = 4
STATE_HOVER = 5


class MouseAction(object):

    def __init__(self):
        self._state = STATE_NONE
        self._state_prev = STATE_NONE
        self._x = 0
        self._y = 0
        self._dx = 0
        self._dy = 0

    def consume(self):
        """Returns the action parameter and sets action state back to STATE_NONE (side-effect).
            Thus an action cannot be consumed multiple times.

        :return: the action parameters
        """
        try:
            return self._state_prev, self._state, self._x, self._y, self._dx, self._dy
        finally:
            self._state_prev = self._state
            self._state = STATE_NONE

    def is_consumed(self):
        return self._state == STATE_NONE

    def is_moved(self):
        return self._dx != 0 and self._dy != 0

    def is_dragged_prev(self):
        return self._state_prev == STATE_DRAGGED

    def is_dragged(self):
        return self._state == STATE_DRAGGED

    def attach_to(self, window, on_mouse_press_fn=None):
        def on_mouse_motion(x, y, dx, dy):
            self._x = x
            self._y = y
            self._dx = dx
            self._dy = dy
            self._state = STATE_MOVED

        def on_mouse_press(x, y, button, modifiers):
            self._x = x
            self._y = y
            self._dx = 0
            self._dy = 0
            self._state = STATE_PRESSED

        def on_mouse_release(x, y, button, modifiers):
            self._x = x
            self._y = y
            self._dx = 0
            self.dy = 0
            self._state = STATE_RELEASED

        def on_mouse_drag(x, y, dx, dy, button, modifiers):
            self._x = x
            self._y = y
            self._dx = dx
            self._dy = dy
            self._state = STATE_DRAGGED

        # Delegate mouse events (to given or defaults)
        window.on_mouse_motion = on_mouse_motion
        if on_mouse_press_fn:
            window.on_mouse_press = on_mouse_press_fn
        else:
            window.on_mouse_press = on_mouse_press
        window.on_mouse_release = on_mouse_release
        window.on_mouse_drag = on_mouse_drag


class MousePointer(MouseCursor):
    """
        Machine controllable mouse widget. The OS mouse is not easily controllable by the environment.
    """

    def __init__(self, x, y, size, color, debug=False):
        self.x = x
        self.y = y
        self.dx = 0
        self.dy = 0
        self.size = size
        self.color = color
        self.state = STATE_NONE  # Consumes and forgets the state
        self.state_prev = STATE_NONE  # Maintains the information of the state
        self.debug = debug
        self.interactables = []
        self.listeners = []

    def on_mouse_action(self, action):
        if action.is_consumed():
            self.state = STATE_NONE  # Set state back to none to prevent repeated drag operations
            return
        self.state_prev, self.state, self.x, self.y, self.dx, self.dy = action.consume()
        if self.state == STATE_MOVED:
            for obj in self.interactables:
                if self.is_over(obj):
                    self.state = STATE_HOVER
        if self.state == STATE_DRAGGED:
            for obj in self.interactables:
                if self.is_over(obj):
                    collides = False
                    for other in self.interactables:
                        collides = obj.collides_proj(other, self.dx, self.dy)
                        if collides:
                            break
                    if not collides:
                        obj.move(self.dx, self.dy)

    def draw(self, x, y):
        pass  # is not drawn by the pyglet framework, but by the viewer

    def is_over(self, obj):
        if obj.x <= self.x <= obj.xw and obj.y <= self.y <= obj.yh:
            return True
        return False

    def set_interactables(self, interactables):
        self.interactables = interactables

    def render(self):
        res = 30
        radius = self.size
        if self.state_prev in [STATE_DRAGGED, STATE_PRESSED]:
            radius *= 2
        glBegin(GL_POLYGON)
        glColor4f(*self.color)
        for i in range(res):
            ang = 2 * math.pi * i / res
            glVertex3f(self.x + math.cos(ang) * radius, self.y + math.sin(ang) * radius, 0)
        glEnd()


# Objects

class Block(object):

    def __init__(self, index, block_length, color=COLOR_ORANGE, decoration=None):
        """
        :param decoration: one of [None, digit, logo] to show on the block
        """
        self.index = index
        self.color = color
        self.block_length = block_length
        self.border_size = 1
        self.x, self.y, self.xw, self.yh = 0, 0, block_length, block_length
        self.x_center, self.y_center = block_length / 2, block_length / 2
        self.decoration = None
        if decoration == "digit":
            self.decoration = pyglet.text.Label(str(self.index),
                                                font_name='Consolas',
                                                font_size=DIGIT_SIZE,
                                                bold=True,
                                                anchor_x="center",
                                                anchor_y="center",
                                                align="center",
                                                x=0, y=0)

    def render(self):
        v = self.get_vertx()
        # Draw block
        glBegin(GL_QUADS)
        glColor4f(*self.color)
        for p in v:  # draw each vertex
            glVertex3f(p[0], p[1], 0)
        glEnd()
        # Draw border
        glLineWidth(self.border_size)
        glBegin(GL_LINE_LOOP)
        glColor4f(*COLOR_BLACK)
        for p in v:
            glVertex3f(p[0], p[1], 0)
        glEnd()
        # Draw decoration
        if self.decoration:
            self.decoration.draw()

        if DEBUG:  # draw center
            glBegin(GL_POLYGON)
            glColor4f(*COLOR_BLACK)
            res = 10
            radius = 3
            for i in range(res):
                ang = 2 * math.pi * i / res
                glVertex3f(self.x_center + math.cos(ang) * radius, self.y_center + math.sin(ang) * radius, 0)
            glEnd()

    def get_vertx(self):
        return [(self.x, self.y), (self.x, self.yh), (self.xw, self.yh), (self.xw, self.y)]

    def move(self, dx, dy):
        self.x += dx
        self.y += dy
        self.xw = self.x + self.block_length
        self.yh = self.y + self.block_length
        if self.decoration:
            self.x_center = self.x + self.block_length / 2
            self.y_center = self.y + self.block_length / 2 + 3  # + offset to "center" a bit better
            self.decoration.x = self.x_center
            self.decoration.y = self.y_center

    def set_location(self, x, y):
        self.x = 0
        self.y = 0
        self.move(x, y)

    def is_over(self, obj):
        if obj.x <= self.xw <= obj.xw and obj.y <= self.yh <= obj.yh:
            return True
        return False

    def collides_proj(self, obj, dx, dy):
        if obj == self:
            return False
        proj_x = self.x + dx
        proj_y = self.y + dy
        if obj.x <= proj_x <= obj.xw and obj.y <= proj_y <= obj.yh:  # bottom-left corner
            return True
        proj_xw = proj_x + self.block_length
        proj_yh = proj_y + self.block_length
        if obj.x <= proj_xw <= obj.xw and obj.y <= proj_yh <= obj.yh:  # upper-right corner
            return True
        if obj.x <= proj_x <= obj.xw and obj.y <= proj_yh <= obj.yh:  # upper-left corner
            return True
        if obj.x <= proj_xw <= obj.xw and obj.y <= proj_y <= obj.yh:  # bottom-right corner
            return True
        return False


class Area(object):

    def __init__(self, w, h, color):
        self.color = color
        self.x = 0
        self.y = 0
        self.xw = self.x + w
        self.yh = self.y + h

    def render(self):
        v = self.get_vertx()
        glBegin(GL_QUADS)
        glColor4f(*self.color)
        for p in v:  # draw each vertex
            glVertex3f(p[0], p[1], 0)
        glEnd()

    def get_vertx(self):
        return [(self.x, self.y), (self.x, self.yh), (self.xw, self.yh), (self.xw, self.y)]


class LabelGeom(object):

    def __init__(self, x, y, text):
        self.label = pyglet.text.Label(text,
                                       font_name='Consolas',
                                       font_size=10,
                                       bold=True,
                                       color=(0, 0, 0, 255),
                                       x=x, y=y)

    def set_location(self, x, y):
        self.label.x = x
        self.label.y = y

    def render(self):
        self.label.draw()


# Locations


class WorldStateLocation(object):
    """
        We assume a quadratic world.
    """

    def __init__(self, side_length):
        self.side_length = side_length

    def create_locations(self, world_state, rel_block_length):
        abs_block_length = rel_block_length * (self.side_length / 2)
        abs_block_length_half = abs_block_length / 2
        # For each pixel position we define a translation scalar
        step_size = abs_block_length / self.side_length
        translation = np.arange(abs_block_length_half, -abs_block_length_half, step=-step_size)
        block_locations = []
        for block_state in world_state:
            block_center = np.array([block_state[0], block_state[2]])
            # Lets assume the state range is [-1,+1] rescale to [0,1]
            block_center_scale = (block_center + 1) / 2
            # Lets assume these are the blocks center coordinates (?)
            block_location = block_center_scale * self.side_length - abs_block_length_half
            # Translate 3d camera-angle to 2d top-view
            block_location = np.around(block_location).astype(int)
            block_location[0] += translation[block_location[0]]
            block_location[1] += translation[block_location[1]]
            # Lets assume discrete pixel space locations
            block_location = np.around(block_location).astype(int)
            # Safety-check: Keep locations in bounds
            block_location[block_location < 0] = 1

            block_locations.append((block_location[0], block_location[1]))
        return block_locations


class BoxLocationGenerator(object):
    """
        Sample locations from within bounds which do not overlap.
    """

    def __init__(self, xmax, ymax, xmin=0, ymin=0):
        self.xmax = xmax
        self.xmin = xmin
        self.ymax = ymax
        self.ymin = ymin

    def __sample_exclusive(self, box_length, board_positions):
        valid_positions = np.argwhere(board_positions)
        valid_indicies = np.arange(len(valid_positions))
        position_idx = np.random.choice(valid_indicies)
        position_x, position_y = valid_positions[position_idx]
        # Mark selected box position (we paint to the right and to the top)
        board_positions[position_x: position_x + box_length, position_y:position_y + box_length] = 0
        # Remove overlapping neighbor positions
        board_positions[position_x - box_length: position_x, position_y:position_y + box_length] = 0  # left
        board_positions[position_x: position_x + box_length, position_y - box_length:position_y] = 0  # bottom
        board_positions[position_x - box_length: position_x, position_y - box_length:position_y] = 0  # bottom-left
        return position_x, position_y

    def sample_exclusive(self, n, box_length):
        board_positions = np.ones(shape=(self.xmax, self.ymax))
        return [self.__sample_exclusive(box_length, board_positions) for _ in range(n)]


class LocationGenerator(object):

    def __init__(self, xmax, ymax, xmin=0, ymin=0):
        self.xmax = xmax
        self.xmin = xmin
        self.ymax = ymax
        self.ymin = ymin

    def create_location(self, placement):
        if "random" in placement:
            mouse_x = np.random.randint(self.xmin, self.xmax)
            mouse_y = np.random.randint(self.ymin, self.ymax)
            return mouse_x, mouse_y
        if "center" in placement:
            mouse_x = self.xmax / 2
            mouse_y = self.ymin + (self.ymax - self.ymin) / 2
            return mouse_x, mouse_y
        raise NotImplementedError()

    def create_random_location_rel(self, placement, obj, distmax=None, offset=None):
        x_max, x_min, y_max, y_min = self.__get_boundaries_rel(placement, obj, distmax, offset)
        # Allow diagonals
        if "left" in placement:
            x_max = obj.x
        if "right" in placement:
            x_min = obj.xw
        if "above" in placement:
            y_min = obj.yh
        if "below" in placement:
            y_max = obj.y
        if "strict" in placement:
            # Narrow to object size (no diagonals)
            if "left" in placement:
                y_min, y_max = obj.y, obj.yh
            if "right" in placement:
                y_min, y_max = obj.y, obj.yh
            if "above" in placement:
                x_min, x_max = obj.x, obj.xw
            if "below" in placement:
                x_min, x_max = obj.x, obj.xw
        return np.random.randint(x_min, x_max), np.random.randint(y_min, y_max)

    def __get_boundaries_rel(self, placement=None, obj=None, distmax=None, offset=None):
        # The whole window
        x_min, x_max = self.xmin, self.xmax
        y_min, y_max = self.ymin, self.ymax
        # The whole window minus offset
        if offset:
            x_min, x_max = self.xmin + offset, self.xmax - offset
            y_min, y_max = self.ymin + offset, self.ymax - offset
        # The object plus/minus offset
        if obj and distmax:
            x_min, x_max = obj.x - distmax, obj.xw + distmax
            y_min, y_max = obj.y - distmax, obj.yh + distmax
        # The object only
        if obj and "on-top" in placement:
            x_min, x_max = obj.x, obj.xw
            y_min, y_max = obj.y, obj.yh
        return x_max, x_min, y_max, y_min


# Environment

class Block2dEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 60,
        'video.res_w': WINDOW_WIDTH,
        'video.res_h': WINDOW_HEIGHT
    }

    def __init__(self, task, params=None, split_name=None, device="cpu"):
        self.task = task
        self.params = params
        self.split_name = split_name
        self.device = device

        self.viewer = None
        self.debug = DEBUG

        self.observation_space = spaces.Box(0, 255, (WINDOW_WIDTH, WINDOW_HEIGHT, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(3)

        self.target_block = -1
        self.score = 0

        self.mouse = None
        self.blocks_by_id = dict()

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: MouseAction, mode="human"):
        """Run one timestep of the environment's dynamics.
            When end of episode is reached, you are responsible for calling `reset()` to reset this environment's state.
            Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the agent

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        step_reward = 0
        if action is not None:
            if self.mouse:
                self.mouse.on_mouse_action(action)
                if action.is_moved():
                    step_reward = -1

        observation = self.render(mode)

        done = False
        if action is not None:
            if self.goal_reached(action):
                step_reward = 1000
                done = True

        self.score += step_reward
        return observation, self.score, done, {}

    def goal_reached(self, action):
        # TODO
        return False

    def __register_objects(self):
        """
            Uses PARAM_NUM_BLOCKS in params to determine the block ids.
        """
        # Create and add blocks at each reset() from world_state
        if self.task[TPARAM_BLOCK_PLACEMENT] == PARAM_WORLD_STATE:
            num_blocks = len(self.params[PARAM_WORLD_STATE])
        else:
            num_blocks = self.params[PARAM_NUM_BLOCKS]

        if isinstance(num_blocks, int):
            num_blocks = list(range(1, num_blocks + 1))

        if TPARAM_BLOCK_COLORS in self.task:
            block_colors = self.task[TPARAM_BLOCK_COLORS]
        else:
            block_colors = dict([(str(block_id), COLOR_GREEN) for block_id in num_blocks])  # monochrom

        self.viewer.geoms = []

        # Create and add background
        self.viewer.add_geom(Area(WINDOW_WIDTH, WINDOW_HEIGHT, COLOR_BACKGROUND))

        # Add checker board lines (for discrete position debugging)
        if DEBUG:
            bin_count = BIN_COUNT
            bin_size = int(WINDOW_HEIGHT / BIN_COUNT)
            for idx in range(bin_count):
                x_offset = bin_size + idx * bin_size
                self.viewer.add_geom(Line(start=(x_offset, 0.0), end=(x_offset, WINDOW_HEIGHT)))
            for idx in range(bin_count):
                y_offset = bin_size + idx * bin_size
                self.viewer.add_geom(Line(start=(0.0, y_offset), end=(WINDOW_WIDTH, y_offset)))

        self.blocks_by_id.clear()
        for block_id in num_blocks:
            self.blocks_by_id[block_id] = Block(block_id,
                                                BLOCK_LENGTH_WINDOW,
                                                color=block_colors[str(block_id)],
                                                decoration=self.task[TPARAM_BLOCK_DECORATION])
        [self.viewer.add_geom(block) for block in self.blocks_by_id.values()]
        # Create and add mouse
        if self.task[TPARAM_MOUSE_PLACEMENT] != "none":
            self.mouse = MousePointer(x=0, y=0, size=MOUSE_SIZE, color=COLOR_BLACK)
            self.mouse.set_interactables(self.blocks_by_id.values())
            self.viewer.add_geom(self.mouse)  # Render mouse (always last as "on-top")

    def get_color(self, block_id):
        block = self.blocks_by_id[block_id]
        return block.color

    def get_bbox(self, block_id):
        """
            MSCOCO: "bbox": [x,y,width,height],
            box coordinates are measured from the top left block corner and are 0-indexed
        """
        block = self.blocks_by_id[block_id]
        return int(block.x), int(block.yh)  # , int(block.block_length), int(block.block_length)

    def get_pos(self, block_id):
        """
            Return the "center" of the bbox, also return the window sizes to relate
        """
        block = self.blocks_by_id[block_id]
        return int(block.x_center), int(block.y_center)  # , WINDOW_WIDTH, WINDOW_HEIGHT

    def get_pos_discrete(self, block_id, bins):
        """
            Returns the discrete position in terms of a "checkered" board
        """
        x, y = self.get_pos(block_id)
        bin_length = WINDOW_HEIGHT / bins
        bin_x = math.floor(x / bin_length)
        bin_y = math.floor(y / bin_length)
        return bin_x, bin_y  # , bins, bins

    def reset(self, mode="human", task=None, params=None):
        """Resets the environment to an initial state and returns an initial observation.

            Note that this function should not reset the environment's random
            number generator(s); random variables in the environment's state should
            be sampled independently between multiple calls to `reset()`. In other
            words, each call of `reset()` should yield an environment suitable for
            a new episode, independent of previous episodes.

        Returns:
            observation (object): the initial observation.
        """
        if task:
            self.task = task
        if params:
            self.params = params

        if self.task[TPARAM_BLOCK_PLACEMENT] == PARAM_WORLD_STATE and self.params[PARAM_WORLD_STATE] is None:
            raise Exception("World states must be given on reset() when block placement is 'world_state'.")

        if self.viewer is None:
            # Notice: This is fine, if we do ALL GL object (e.g. labels) creation afterwards!
            # The other environments do not have these problems, because they use onetime geoms only.
            self.viewer = rendering.Viewer(WINDOW_WIDTH, WINDOW_HEIGHT)
            self.viewer.window.set_mouse_visible(False)
            self.viewer.window.set_visible(mode == "human")

        self.__register_objects()

        # Apply configuration and initial placements
        if self.mouse:
            sampler = LocationGenerator(WINDOW_WIDTH, WINDOW_HEIGHT)
            self.mouse.x, self.mouse.y = sampler.create_location([self.task[TPARAM_MOUSE_PLACEMENT]])

        if self.task[TPARAM_BLOCK_PLACEMENT] == "random":
            sampler = BoxLocationGenerator(WINDOW_WIDTH - BLOCK_LENGTH_WINDOW,
                                           WINDOW_HEIGHT - BLOCK_LENGTH_WINDOW)
            block_locations = sampler.sample_exclusive(len(self.blocks_by_id), BLOCK_LENGTH_WINDOW)
        elif self.task[TPARAM_BLOCK_PLACEMENT] == "center":
            # Note: This only "works" with single blocks
            center_x, center_y = WINDOW_WIDTH / 2 - BLOCK_LENGTH_WINDOW / 2, WINDOW_HEIGHT / 2 - BLOCK_LENGTH_WINDOW / 2
            block_locations = [(center_x, center_y) for _ in self.blocks_by_id]
        elif self.task[TPARAM_BLOCK_PLACEMENT] == PARAM_WORLD_STATE:
            world_states_location = WorldStateLocation(WINDOW_WIDTH)
            world_state = self.params[PARAM_WORLD_STATE]
            block_locations = world_states_location.create_locations(world_state.cpu().numpy(), BLOCK_LENGTH)
        else:
            raise NotImplementedError("Unknown block placement: " + self.task[TPARAM_BLOCK_PLACEMENT])

        for block_id, block_location in zip(self.blocks_by_id, block_locations):
            if block_location:
                block_x, block_y = block_location
                self.blocks_by_id[block_id].set_location(block_x, block_y)
            # Other blocks might not have fit on the board anymore

        return self.step(action=None, mode=mode)[0]

    def render(self, mode='human'):
        """
            The RGB array has shape (H, W, C)
        """
        assert mode in ["human", "state_pixels", "rgb_array"]

        # Add pos to checker board
        if DEBUG:
            for block_id in self.blocks_by_id:
                block = self.blocks_by_id[block_id]
                pos_discrete = self.get_pos_discrete(block_id, bins=BIN_COUNT)
                block.decoration.text = "%s|%s" % (pos_discrete[0], pos_discrete[1])
        return self.viewer.render(return_rgb_array=True if mode == "rgb_array" else False)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    @staticmethod
    def run_interactive(task, params):
        game = Block2dEnv(task, params)
        game.reset()
        action = MouseAction()
        action.attach_to(game.viewer.window)
        is_open = True
        while is_open:
            _, _, done, _ = game.step(action)
            if done:
                game.reset()
            is_open = game.viewer.isopen
        game.close()

    @staticmethod
    def run_interactive_plot(num_blocks=16):
        from matplotlib import pyplot as plt
        game = Block2dEnv(num_blocks)
        game.reset()
        action = MouseAction()
        action.attach_to(game.viewer.window)
        is_open = True
        fig, ax = plt.subplots(1, 1)
        image = game.render(mode="rgb_array")
        im = ax.imshow(image)
        while is_open:
            image, _, done, _ = game.step(action, mode="rgb_array")
            if done:
                game.reset()
            im.set_data(image)
            fig.canvas.draw_idle()
            plt.pause(.1)
            is_open = game.viewer.isopen
        game.close()


if __name__ == "__main__":
    Block2dEnv.run_interactive(task={
        TPARAM_MOUSE_PLACEMENT: "center",
        TPARAM_BLOCK_PLACEMENT: "random",
        TPARAM_BLOCK_DECORATION: "digit",
        TPARAM_BLOCK_COLORS: {
            "1": [1.0, 0.0, 1.0, 1.0],
            "2": [0.9, 0.1, 0.0, 1.0],
            "3": [0.8, 0.2, 1.0, 1.0],
            "4": [0.7, 0.3, 0.0, 1.0],
            "5": [0.6, 0.4, 1.0, 1.0],
            "6": [0.5, 0.5, 0.0, 1.0],
            "7": [0.4, 0.6, 1.0, 1.0],
            "8": [0.3, 0.7, 0.0, 1.0],
            "9": [0.2, 0.8, 1.0, 1.0]
        }
    }, params={
        PARAM_NUM_BLOCKS: 4
    })
