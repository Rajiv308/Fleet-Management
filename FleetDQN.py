from tkinter import *
from tkinter import ttk
import threading
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from PIL import Image, ImageTk
import tkinter.messagebox

from tensorflow import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers.core import Dense, Activation
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from keras.layers.advanced_activations import PReLU
import json

# import keras.backend.tensorflow_backend as backend
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.callbacks import TensorBoard
import tensorflow as tf
from collections import deque
import time
import random
from tqdm import tqdm
import os
from PIL import Image
import cv2


class App:
    def __init__(self, master):
        self.master = master

        #        grid map setting
        self.grid_origx = 500
        # self.grid_origy = 20
        self.grid_origy = 40
        self.grid_columnNum = 8
        self.grid_rowNum = 8
        self.grid_UNIT = 90
        self.maze_size = self.grid_columnNum * self.grid_rowNum
        #        define total training episodes
        self.episode = 10_000
        #        define number of tests to run
        self.tests = 100
        #        set a small amount of delay (second) to make sure tkinter works properly
        #        if want to have a slower visualization for testing, set the delay to larger values
        # self.timeDelay = 0.1
        self.timeDelay = 0.0001

        self.epsilon = 0.9
        self.decay_rate = 2 * self.epsilon / self.episode

        #       other initialization
        self.n_actions = 4
        self.outline = 'black'
        self.fill = None
        self.item_type = 0
        self.learning = False
        self.itemsNum = 0
        self.Qtable_origx = self.grid_origx + 20 + (self.grid_columnNum + 1) * self.grid_UNIT
        self.Qtable_origy = self.grid_origy
        self.grid_origx_center = self.grid_origx + self.grid_UNIT / 2
        self.grid_origy_center = self.grid_origy + self.grid_UNIT / 2
        self.Qtable_gridIndex_dict = {}
        self.show_q_table = pd.DataFrame(columns=list(range(self.n_actions)), dtype=np.float64)
        self.origDist = 10

        self.agentCentre = np.array([[190, 250], [290, 250], [390, 250]])
        self.targetCentre = self.agentCentre + np.array([[0, self.grid_UNIT + self.origDist],
                                                         [0, self.grid_UNIT + self.origDist],
                                                         [0, self.grid_UNIT + self.origDist]])
        self.ObstacleCentre1 = np.array([[725, 535], [725, 355], [635, 715]])

        self.ObstacleCentre2 = np.array([[905, 265], [545, 265], [995, 625]])
        self.StoreCentre = np.array([])

        self.agentPosition_list = []
        self.targetPosition_list = []
        self.ObstaclePosition_list = []
        self.StorePosition_list = []

        self.targetItemIndex = []
        self.agentItemIndex = []
        self.obstacleItemIndex = []
        self.storeItemIndex = []

        self.AllItemsOrigPosition_list = {}
        self.createMarkT = []
        self.createMarkS = []
        self.createMarkobs = []
        self.points = []

        self.selected_agents = []
        self.selected_agent_position = []
        self.selected_Obstacles = []
        self.selected_Obstacles_position = []
        self.selected_targets = []

        self.store = None
        self.store_visit = False

        self.arrivedObs_id = 0
        self.arrivedTar_id = 0
        self.init_widgets()
        self.temp_item = None
        self.temp_items = []
        self.choose_item = None
        self.created_line = []
        self.lines = []

    def init_widgets(self):

        self.cv = Canvas(root, background='white')
        item.canvas_copy(self.cv, self.grid_origx, self.grid_origy, self.grid_columnNum, self.grid_rowNum,
                         self.grid_UNIT)
        self.cv.pack(fill=BOTH, expand=True)
        # bind events of dragging with mouse
        self.cv.bind('<B1-Motion>', self.move)
        self.cv.bind('<ButtonRelease-1>', self.move_end)
        self.cv.bind("<Button-1>", self.leftClick_handler)

        # bind events of double-left-click
        self.cv.bind("<Button-3>", self.rightClick_handler)
        f = ttk.Frame(self.master)
        f.pack(fill=X)
        self.bns = []

        # initialize buttons
        for i, lb in enumerate(('Reset', 'Start trainning', 'Close', 'Save', 'Start Running')):
            bn = Button(f, text=lb, command=lambda i=i: self.choose_type(i))
            bn.pack(side=LEFT, ipadx=8, ipady=5, padx=5)
            self.bns.append(bn)
        # self.bns[self.item_type]['relief'] = SUNKEN

        # initialize agent, warehouses and obstacles positions
        self.agentPosition_list = self.agentCentre.tolist()
        self.targetPosition_list = self.targetCentre.tolist()
        self.ObstaclePosition_list1 = self.ObstacleCentre1.tolist()
        self.ObstaclePosition_list2 = self.ObstacleCentre2.tolist()
        self.ObstaclePosition_list = self.ObstaclePosition_list1 + self.ObstaclePosition_list2
        self.create_items()
        self.StorePosition_list = self.StoreCentre.tolist()
        self.itemsNum = self.targetCentre.shape[0] + self.ObstacleCentre1.shape[0] + self.ObstacleCentre2.shape[0] + \
                        self.agentCentre.shape[0] + self.StoreCentre.shape[0]
        R = self.grid_UNIT
        self.cv.create_text(self.agentCentre[0][0] - R - 20, self.agentCentre[0][1],
                            text="Agent:", font=('Courier', 18))
        self.cv.create_text(self.targetCentre[0][0] - R - 20, self.targetCentre[0][1],
                            text="Warehouse:", font=('Courier', 18))
        self.cv.create_text(self.grid_origx + 250, self.grid_origy - 20, text="Multi agent Deep Q-Learning Simulation",
                            font=('Times', 25), fill='red')
        self.cv.create_text(self.grid_origx + 252, self.grid_origy - 22, text="Multi agent Deep Q-Learning Simulation",
                            font=('Times', 25), fill='green')

        # draw grids
        self.create_grids(self.grid_origx, self.grid_origy, self.grid_columnNum, self.grid_rowNum, self.grid_UNIT)

        for i in range(0, self.grid_rowNum):
            for j in range(0, self.grid_columnNum):
                x = i * self.grid_UNIT + self.grid_origx_center
                y = j * self.grid_UNIT + self.grid_origy_center
                rowIndex = (y - self.grid_origy_center) / self.grid_UNIT
                columnIndex = (x - self.grid_origx_center) / self.grid_UNIT
                self.Qtable_gridIndex_dict[(x, y)] = rowIndex * self.grid_columnNum + columnIndex

        print(self.Qtable_gridIndex_dict)

    def choose_type(self, i):
        """
        function of clicking different button
        """
        for b in self.bns:
            b['relief'] = RAISED
        self.bns[i]['relief'] = SUNKEN
        self.item_type = i
        if self.item_type == 1:
            #            start training
            self.start()
            self.bns[i]['relief'] = RAISED

        elif self.item_type == 2:
            #            close simulation tool
            os._exit(0)

        elif self.item_type == 3:
            #           save q_table
            # temp_t = str(self.selected_targets_position) +"_" + str(len(self.selected_agents))
            time = datetime.datetime.now()
            self.agent_model.target_model.save('models/DQNModel_' + str(len(self.selected_targets)) + 'Targets_V6.h5')
            print("SAVED!!!")
            self.labelHello = Label(self.cv, text="table saved!!", font=("Helvetica", 10), width=10, fg="red",
                                    bg="white")
            self.labelHello.place(x=350, y=750, anchor=NW)
            pass
        elif self.item_type == 0:
            self.button_reset()
            self.bns[i]['relief'] = RAISED
        elif self.item_type == 4:
            #            start running tests
            self.start(running=True)
            pass
        elif self.item_type == 5:
            self.restart()

    def create_items(self):
        self.create_agentItems()
        self.agentItemIndex = [1, len(self.agentPosition_list)]

        self.create_targetItems()
        self.targetItemIndex = [self.agentItemIndex[1] + 1,
                                self.agentItemIndex[1] + len(self.targetPosition_list)]

        self.create_ObsItems()
        self.obstacleItemIndex = [self.targetItemIndex[1] + 1,
                                  self.targetItemIndex[1] + len(self.ObstaclePosition_list)]

        self.create_Store()
        self.StorePosition_list = self.StoreCentre.tolist()
        self.storeItemIndex = [self.obstacleItemIndex[1] + 1,
                               self.obstacleItemIndex[1] + len(self.StorePosition_list)]
        item.classify(self.agentItemIndex, self.targetItemIndex, self.obstacleItemIndex, self.storeItemIndex)
        self.AllItemsOrigPosition_list = item.AllItemsOrigPosition_list

    def create_ObsItems(self):
        self.cv.obstacles = ['obs5.jpg', 'obs7.jpg', 'obs8.jpg']
        self.cv.obstacles += self.cv.obstacles
        self.cv.arriveObsImage = ['obs5_car.jpg']
        self.obstacles = []

        for i, j in zip(self.cv.obstacles, self.ObstaclePosition_list):
            self.obstacles.append(item(i, j))

        # arriving picture
        # self.arrivedObs = item('obs5_car.jpg', (2000,2000))

    def create_targetItems(self):
        self.cv.targets = ['warehouse4_1.jpg', 'warehouse3.jpg', 'warehouse4_2.jpg']
        self.cv.arriveImage = ['warehouse3_car.jpg']
        self.targets = []

        for i, j in zip(self.cv.targets, self.targetPosition_list):
            self.targets.append(item(i, j))

        # arriving picture
        # self.arrivedTar = item('warehouse3_car.jpg', (2000,2000))

    def create_agentItems(self):
        self.cv.cars = ['car9.jpg', 'car2.jpg', 'car8.jpg']
        w_box, h_box = self.grid_UNIT, self.grid_UNIT
        self.agents = []

        for i, j in zip(self.cv.cars, self.agentPosition_list):
            self.agents.append(item(i, j))

    def create_Store(self):
        while True:
            new_loc = [
                random.randrange(self.grid_origx_center, self.grid_rowNum * self.grid_UNIT + self.grid_origx_center,
                                 self.grid_UNIT),
                random.randrange(self.grid_origy_center, self.grid_columnNum * self.grid_UNIT + self.grid_origy_center,
                                 self.grid_UNIT)]
            if new_loc not in self.AllItemsOrigPosition_list.values():
                break
        # self.store = item('store.jpg', new_loc)
        self.StoreCentre = np.array([new_loc])

    def create_grids(self, origx, origy, column, row, UNIT):
        # create grids
        for c in range(origx, origx + (column + 1) * UNIT, UNIT):
            x0, y0, x1, y1 = c, origy, c, origy + row * UNIT
            self.cv.create_line(x0, y0, x1, y1, width=2)
        for r in range(origy, origy + (row + 1) * UNIT, UNIT):
            x0, y0, x1, y1 = origx, r, origx + row * UNIT, r
            self.cv.create_line(x0, y0, x1, y1, width=2)

    def get_loc_list(self, selected):
        loc_list = []
        for a in selected:
            loc_list.append(self.cv.coords(a))
        return loc_list

    def start(self, running=False):
        """
        initialization for training process
        """
        self.selected_agents, self.selected_targets, self.selected_Obstacles, self.selected_store = item.check_selections()

        self.selected_agent_position = self.get_loc_list(self.selected_agents)
        self.selected_targets_position = self.get_loc_list(self.selected_targets)
        self.selected_Obstacles_position = self.get_loc_list(self.selected_Obstacles)
        self.arrivedTar = []
        self.arrivedObs = []
        for i in range(len(self.selected_agents)):
            self.arrivedTar.append(item('warehouse3_car.jpg', (2000, 2000)))
            self.arrivedObs.append(item('obs5_car.jpg', (2000, 2000)))

        if len(self.selected_agents) == 0:
            tkinter.messagebox.showinfo("INFO", "Please choose ONE agent for training！")
        elif len(self.selected_targets) == 0:
            tkinter.messagebox.showinfo("INFO", "Please choose ONE target for training！")
        else:
            if running:
                self.t = threading.Timer(self.timeDelay, self.running)
                self.t.start()
                self.learning = True
            else:
                self.t = threading.Timer(self.timeDelay, self.update)
                self.t.start()
                self.learning = True

    def update(self):
        self.win_history = []
        episode = 0
        self.agent_model = DQNAgent()
        self.done = False
        observations = []
        self.current_loc = self.selected_agent_position
        agent_reward_list = []
        agent_done_list = []
        visited = []
        self.epsilon = 0.9
        hit = False
        t = 0
        for i, a in enumerate(self.selected_agents):
            # self.agent_models.append(DQNAgent())
            observations.append(self.get_normal_array())
            index = int(self.Qtable_gridIndex_dict[tuple(self.cv.coords(a))])
            x, y = self.get_indices(index)
            observations[i][x][y] = 1
            visited.append(set())
            visited[i].add(tuple(self.cv.coords(a)))
            agent_reward_list.append(0)
            agent_done_list.append(False)
        loss = 0.0
        start_time = datetime.datetime.now()
        loss_list = []
        avg_reward_list = []
        action = -1
        total_reward = 0
        total_reward_list = []
        self.labelHello = Label(self.cv, text="start training!", font=("Helvetica", 10), width=10, fg="red", bg="white")
        self.labelHello.place(x=200, y=750, anchor=NW)
        stepCount = 0
        while True:
            self.labelHello = Label(self.cv, text="episode: %s" % str(episode), font=("Helvetica", 10), width=10,
                                    fg="blue", bg="white")
            self.labelHello.place(x=200, y=550, anchor=NW)
            self.render()
            check = False
            for i, a in enumerate(self.selected_agents):
                if agent_done_list[i]:
                    continue
                check = True
                action = self.action(self.agent_model, observations[i])
                self.render()
                s = self.cv.coords(a)
                new_state = self.step(a, action)
                reward, done = self.reward(self.cv.coords(a))
                self.current_loc[i] = self.cv.coords(a)

                if s == self.current_loc[i]:
                    reward -= 30

                if reward <= -300:
                    hit = True

                if tuple(self.current_loc[i]) in visited[i]:
                    reward -= 10
                visited[i].add(tuple(self.current_loc[i]))

                agent_reward_list[i] += reward
                agent_done_list[i] = done

                self.agent_model.update_memory(
                    (observations[i].reshape(8, 8, 1), action, reward, new_state.reshape(8, 8, 1), done))

                observations[i] = new_state

                if agent_reward_list[i] <= -300:
                    hit = True
                    agent_done_list[i] = True

            if not (False in agent_done_list) and not (hit):
                self.win_history.append(1)
                self.done = True

            elif not (check):
                self.win_history.append(0)
                self.done = True

            self.agent_model.train(self.done)

            stepCount += 1

            if self.done:
                episode += 1
                for i in range(len(self.selected_agents)):
                    total_reward += agent_reward_list[i]
                    agent_reward_list[i] = 0
                    agent_done_list[i] = False
                    visited[i] = set()

                if episode > self.episode:
                    self.reset(end=True)
                    break

                else:
                    time.sleep(self.timeDelay)
                    self.reset(agent=False)
                    tracker = 0
                    for i in self.agents:
                        if i.id in self.selected_agents:
                            self.current_loc[tracker] = i.random()
                            observations[tracker] = self.get_normal_array()
                            index = int(self.Qtable_gridIndex_dict[tuple(self.current_loc[tracker])])
                            x, y = self.get_indices(index)
                            observations[tracker][x][y] = 1
                            tracker += 1
                    self.selected_targets_position = self.get_loc_list(self.selected_targets)

                    dt = datetime.datetime.now() - start_time
                    t = self.format_time(dt.total_seconds())

                    total_reward_list.append(total_reward)
                    if len(total_reward_list) > 100:
                        avg_reward = sum(total_reward_list[-100:]) / 100
                        avg_reward_list.append(avg_reward)
                        template = "Episode: {:03d}/{:d} | StepCount: {:d} | Win: {:b} | Win rate: {:.3f} | Total rewards: {:.3f} | Average rewards: {:.3f} | time: {}"
                        print(template.format(episode, self.episode, stepCount, bool(self.win_history[-1]),
                                              sum(self.win_history) / len(self.win_history),
                                              total_reward, avg_reward, t))
                    else:
                        template = "Episode: {:03d}/{:d} | StepCount: {:d} | Win: {:b} | Win rate: {:.3f} | Total rewards: {:.3f} | time: {}"
                        print(template.format(episode, self.episode, stepCount, bool(self.win_history[-1]),
                                              sum(self.win_history) / len(self.win_history),
                                              total_reward, t))
                    stepCount = 0
                    total_reward = 0
                    self.epsilon_decay()
                    self.done = False
                    hit = False

        print('training over!')
        self.labelHello = Label(self.cv, text="Training Over!", font=("Helvetica", 10), width=10, fg="red",
                                bg="white")
        self.labelHello.place(x=200, y=750, anchor=NW)
        print("total_win_rate", sum(self.win_history) / len(self.win_history))
        print("total_time", t)
        print("average rewards per episode", sum(total_reward_list) / len(total_reward_list))
        self.learning = False

        # In case you would like to see how the model performs
        # you can uncomment the following code to see the graphs

        # plt.figure()
        # plt.title('Rewards per Episode')
        # plt.xlabel('Episode number')
        # plt.ylabel('Rewards')
        # plt.plot(total_reward_list)
        # plt.show()

        # plt.figure()
        # plt.title('Average Rewards over 100 Episode')
        # plt.xlabel('Episode number')
        # plt.ylabel('Rewards')
        # plt.plot(avg_reward_list)
        # plt.show()

    def action(self, agent, state):
        if np.random.random() > self.epsilon:
            action = np.argmax(agent.get_qs(state.reshape(8, 8, 1)))

        else:
            # Get random action
            action = np.random.randint(0, 4)
        return action

    def step(self, agent, action):
        UNIT = self.grid_UNIT
        loc = np.array(self.cv.coords(agent))
        origin = np.array([self.grid_origx, self.grid_origy])
        s = (loc - origin).tolist()
        base_action = np.array([0, 0])
        if action == 0:  # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # down
            if s[1] < (self.grid_rowNum - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # right
            if s[0] < (self.grid_columnNum - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:  # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        s_ = [loc[0] + base_action[0], loc[1] + base_action[1]]

        agent_obj = item.get_item(agent)
        agent_obj.show(s_)

        new_state = self.get_normal_array()
        index = int(self.Qtable_gridIndex_dict[tuple(self.cv.coords(agent))])
        x, y = self.get_indices(index)
        new_state[x][y] = 1

        return new_state

    def reward(self, s_):
        done = False
        if s_ in self.selected_targets_position:
            reward = 150
            self.arrivedTar[self.arrivedTar_id].show(s_)
            self.arrivedTar_id += 1
            done = True

        elif s_ in self.selected_Obstacles_position:
            reward = -300
            self.arrivedObs[self.arrivedObs_id].show(s_)
            self.arrivedObs_id += 1
            done = True

        else:
            reward = -1
        return reward, done

    def running(self):
        self.win_history = []
        episode = 0
        self.agent_models = DQNAgent()
        self.done = False
        observations = []
        self.current_loc = self.selected_agent_position
        agent_reward_list = []
        agent_done_list = []
        steps_stuck = []
        t = 0
        for i, a in enumerate(self.selected_agents):
            observations.append(self.get_normal_array())
            index = int(self.Qtable_gridIndex_dict[tuple(self.cv.coords(a))])
            x, y = self.get_indices(index)
            observations[i][x][y] = 1
            agent_reward_list.append(0)
            agent_done_list.append(False)
            steps_stuck.append(0)
        start_time = datetime.datetime.now()
        avg_reward_list = []
        total_reward = 0
        total_reward_list = []
        self.labelHello = Label(self.cv, text="start running!", font=("Helvetica", 10), width=10, fg="red", bg="white")
        self.labelHello.place(x=200, y=750, anchor=NW)
        stepCount = 0
        while True:
            self.labelHello = Label(self.cv, text="Test: %s" % str(episode), font=("Helvetica", 10), width=10,
                                    fg="blue", bg="white")
            self.labelHello.place(x=200, y=550, anchor=NW)
            stepCount += 1
            lost = False
            for i, a in enumerate(self.selected_agents):
                if agent_done_list[i]:
                    continue

                action = self.predict_action(self.agent_models, observations[i].reshape(8, 8, 1))
                s = self.cv.coords(a)
                self.render()
                new_state = self.new_step(a, action)
                reward, done = self.new_reward(self.cv.coords(a))
                self.current_loc[i] = self.cv.coords(a)
                agent_done_list[i] = done

                agent_reward_list[i] += reward
                observations[i] = new_state

                if s == self.current_loc[i]:
                    steps_stuck[i] += 1

                if steps_stuck[i] > 3 or stepCount > 25:
                    print('STUCK')
                    lost = True

                if not agent_done_list[i]:
                    line = self.cv.create_line(s[0], s[1],
                                               self.current_loc[i][0], self.current_loc[i][1],
                                               fill='red',
                                               arrow=LAST,
                                               arrowshape=(10, 20, 8),
                                               dash=(4, 4)
                                               )
                    self.lines.append(line)

                if agent_reward_list[i] <= -300:
                    lost = True

            if not (False in agent_done_list) and not (lost):
                self.win_history.append(1)
                self.done = True

            if stepCount > 25 or lost:
                self.win_history.append(0)
                self.done = True

            if self.done:
                episode += 1
                for i in range(len(self.selected_agents)):
                    total_reward += agent_reward_list[i]
                    agent_reward_list[i] = 0
                    agent_done_list[i] = False

                if episode >= self.tests:
                    self.reset(end=True)
                    break
                else:
                    time.sleep(self.timeDelay)
                    self.reset(agent=False)
                    tracker = 0
                    for i in self.agents:
                        if i.id in self.selected_agents:
                            self.current_loc[tracker] = i.random()
                            observations[tracker] = self.get_normal_array()
                            index = int(self.Qtable_gridIndex_dict[tuple(self.current_loc[tracker])])
                            x, y = self.get_indices(index)
                            observations[tracker][x][y] = 1
                            steps_stuck[tracker] = 0
                            tracker += 1
                    self.selected_targets_position = self.get_loc_list(self.selected_targets)

                    dt = datetime.datetime.now() - start_time
                    t = self.format_time(dt.total_seconds())

                    total_reward_list.append(total_reward)
                    if len(total_reward_list) > 100:
                        avg_reward = sum(total_reward_list[-100:]) / 100
                        avg_reward_list.append(avg_reward)
                        template = "Episode: {:03d}/{:d} | StepCount: {:d} | Win: {:b} | Win rate: {:.3f} | Total rewards: {:.3f} | Average rewards: {:.3f} | time: {}"
                        print(template.format(episode, self.tests, stepCount, bool(self.win_history[-1]),
                                              sum(self.win_history) / len(self.win_history),
                                              total_reward, avg_reward, t))
                    else:
                        template = "Episode: {:03d}/{:d} | StepCount: {:d} | Win: {:b} | Win rate: {:.3f} | Total rewards: {:.3f} | time: {}"
                        print(template.format(episode, self.tests, stepCount, bool(self.win_history[-1]),
                                              sum(self.win_history) / len(self.win_history),
                                              total_reward, t))
                    stepCount = 0
                    total_reward = 0
                    self.epsilon_decay()
                    self.done = False

    def predict_action(self, model, state):
        action = np.argmax(model.target_model.predict(np.array(state).reshape(-1, *state.shape) / 3)[0])
        return action

    def new_reward(self, s_):
        done = False
        if s_ in self.selected_targets_position:
            reward = 50
            self.arrivedTar[self.arrivedTar_id].show(s_)
            self.arrivedTar_id += 1
            done = True

        elif s_ in self.selected_Obstacles_position:
            reward = -300
            self.arrivedObs[self.arrivedObs_id].show(s_)
            self.arrivedObs_id += 1
            done = False

        else:
            reward = 0
        return reward, done

    def new_step(self, agent, action):
        UNIT = self.grid_UNIT
        loc = np.array(self.cv.coords(agent))
        origin = np.array([self.grid_origx, self.grid_origy])
        s = (loc - origin).tolist()
        base_action = np.array([0, 0])
        if action == 0:  # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # down
            if s[1] < (self.grid_rowNum - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # right
            if s[0] < (self.grid_columnNum - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:  # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        s_ = [loc[0] + base_action[0], loc[1] + base_action[1]]
        if not (s_ in self.current_loc and s_ != self.cv.coords(agent) and s_ not in self.selected_targets_position):
            agent_obj = item.get_item(agent)
            agent_obj.show(s_)

        new_state = self.get_normal_array()
        index = int(self.Qtable_gridIndex_dict[tuple(self.cv.coords(agent))])
        x, y = self.get_indices(index)
        new_state[x][y] = 1

        return new_state

    def epsilon_decay(self):
        if self.epsilon > 0:
            self.epsilon -= self.decay_rate

    def rightClick_handler(self, event):
        # to return to original location
        if self.choose_item is not None:
            t = self.choose_item
            self.cv.coords(t, self.AllItemsOrigPosition_list[t])
            self.itemOrigPosition = []

    def leftClick_handler(self, event):
        """
        bind events of choosing warehouse
        """

        if self.learning:
            print("Learing on going!")
        else:
            for i in range(1, self.itemsNum + 1):
                position = self.cv.coords(i)
                R = self.grid_UNIT / 2
                p = [position[0] - R, position[1] - R, position[0] + R, position[1] + R]
                if p[0] <= event.x <= p[2] and p[1] <= event.y <= p[3]:
                    t = i
                    self.choose_item_handler(event, t)

    def choose_item_handler(self, event, t):
        self.choose_item = t

    def move(self, event):
        if self.choose_item is not None:
            t = self.choose_item
            self.cv.coords(t, event.x, event.y)

    def move_end(self, event):
        if self.choose_item is not None:
            t = self.choose_item
            item.adjust_items_into_grids(event, t)
            self.choose_item = None

    def delete_item(self, event):
        if self.choose_item is not None:
            self.cv.delete(self.choose_item)

    def reset(self, end=False, agent=True, target=True):
        """
        reset the agent to a random valid location
        """
        if self.lines != []:
            for line in self.lines:
                self.cv.delete(line)

        for Tmarks in range(self.arrivedTar_id):
            self.arrivedTar[Tmarks].show((2000, 2000))
        self.arrivedTar_id = 0

        for Smarks in self.createMarkS:
            self.cv.delete(Smarks)
            self.createMarkS.pop(0)

        for obs in range(self.arrivedObs_id):
            self.arrivedObs[obs].show((2000, 2000))
        self.arrivedObs_id = 0

        if not end:
            if agent:
                for a in self.agents:
                    if a.id in self.selected_agents:
                        a.random()
            if target:
                for b in self.targets:
                    if b.id in self.selected_targets:
                        b.random()

        if end:
            self.learning = False
            item.reset_obj()

        self.store_visit = False

    def button_reset(self):
        time.sleep(self.timeDelay)
        self.reset(end=True)

    def get_normal_array(self):
        input = np.empty((8, 8))
        input.fill(0)
        for b in self.selected_targets:
            index = int(self.Qtable_gridIndex_dict[tuple(self.cv.coords(b))])
            x, y = self.get_indices(index)
            input[x][y] = 2
        for c in self.selected_Obstacles:
            index = int(self.Qtable_gridIndex_dict[tuple(self.cv.coords(c))])
            x, y = self.get_indices(index)
            input[x][y] = 3
        return input

    def get_indices(self, i):
        key_list = list(self.Qtable_gridIndex_dict.keys())
        val_list = list(self.Qtable_gridIndex_dict.values())
        ind = val_list.index(i)
        indices = key_list[ind]
        y = int((indices[0] - self.grid_origx_center) / self.grid_UNIT)
        x = int((indices[1] - self.grid_origy_center) / self.grid_UNIT)
        return x, y

    def render(self):
        time.sleep(self.timeDelay)

    def format_time(self, seconds):
        if seconds < 400:
            s = float(seconds)
            return "%.1f seconds" % (s,)
        elif seconds < 4000:
            m = seconds / 60.0
            return "%.2f minutes" % (m,)
        else:
            h = seconds / 3600.0
            return "%.2f hours" % (h,)


class item:
    w_box = 90
    h_box = 90
    items_list = {}
    AllItemsOrigPosition_list = {}
    AllItemsCurrentPosition_list = {}
    selected_agents = []
    selected_targets = []
    selected_store = None
    selected_obstacles = []

    @classmethod
    def canvas_copy(cls, canvas, x, y, col, row, unit):
        cls.cv = canvas
        cls.grid_origx = x
        cls.grid_origy = y
        cls.grid_columnNum = col
        cls.grid_rowNum = row
        cls.grid_UNIT = unit
        cls.grid_origx_center = x + unit / 2
        cls.grid_origy_center = y + unit / 2
        cls.w_box = unit
        cls.h_box = unit

    @classmethod
    def adjust_items_into_grids(cls, event, id):
        position = cls.cv.coords(id)
        centerX = position[0]
        centerY = position[1]
        Grids_X0 = cls.grid_origx
        Grids_X1 = cls.grid_origx + (cls.grid_columnNum) * cls.grid_UNIT
        Grids_Y0 = cls.grid_origy
        Grids_Y1 = cls.grid_origy + (cls.grid_rowNum) * cls.grid_UNIT
        if (centerX in range(Grids_X0, Grids_X1)) and (centerY in range(Grids_Y0, Grids_Y1)):
            columnIndex = math.floor((centerX - Grids_X0) / cls.grid_UNIT)
            rowIndex = math.floor((centerY - Grids_Y0) / cls.grid_UNIT)
            adjustedX0 = Grids_X0 + columnIndex * cls.grid_UNIT + cls.grid_UNIT / 2
            adjustedY0 = Grids_Y0 + rowIndex * cls.grid_UNIT + cls.grid_UNIT / 2
            cls.cv.coords(id, adjustedX0, adjustedY0)
        else:
            # return to original position if not drag near grids
            cls.cv.coords(id, cls.AllItemsOrigPosition_list[id])

    @classmethod
    def classify(cls, agent_ind, target_ind, obs_ind, store_ind):
        cls.category = {0: range(agent_ind[0], agent_ind[1] + 1),
                        1: range(target_ind[0], target_ind[1] + 1),
                        2: range(obs_ind[0], obs_ind[1] + 1),
                        3: range(store_ind[0], store_ind[1] + 1)}

    @classmethod
    def reset_obj(cls):
        for i, j in item.AllItemsOrigPosition_list.items():
            p = cls.cv.coords(i)
            if p[0] >= cls.grid_origx and p[1] >= cls.grid_origy:
                cls.cv.coords(i, j)

    @classmethod
    def check_selections(cls):
        cls.selected_agents = []
        cls.selected_targets = []
        cls.selected_store = []
        cls.selected_obstacles = []
        for i in range(1, len(cls.items_list) + 1):

            p = cls.cv.coords(i)

            if cls.grid_origx <= p[0] <= 1220 and cls.grid_origy <= p[1] <= 1220:
                if i in cls.category.get(0):
                    cls.selected_agents.append(i)
                elif i in cls.category.get(1):
                    cls.selected_targets.append(i)
                elif i in cls.category.get(2):
                    cls.selected_obstacles.append(i)
                elif i in cls.category.get(3):
                    cls.selected_store.append(i)
        return cls.selected_agents, cls.selected_targets, cls.selected_obstacles, cls.selected_store

    @classmethod
    def get_item(cls, id):
        return cls.items_list[id]

    def __init__(self, file, location=None, show=True):
        pil_image = Image.open(file)
        w, h = pil_image.size
        pil_image_resized = self.resize(pil_image)
        tk_image = ImageTk.PhotoImage(pil_image_resized)
        self.img = tk_image
        if show:
            self.id = item.cv.create_image(location, image=self.img)
            item.items_list[self.id] = self
            item.AllItemsOrigPosition_list[self.id] = item.cv.coords(self.id)
            item.AllItemsCurrentPosition_list[self.id] = item.cv.coords(self.id)

    def resize(self, pil_image):
        '''''
      resize a pil_image
      '''
        return pil_image.resize((self.w_box, self.h_box), Image.ANTIALIAS)

    def show(self, location):
        item.cv.coords(self.id, location)
        item.AllItemsCurrentPosition_list[self.id] = location

    def remove(self):
        self.cv.delete(self.id)

    def locate(self):
        return item.cv.coords(self.id)

    def random(self, place=True):
        while True:
            new_loc = [
                random.randrange(self.grid_origx_center, self.grid_rowNum * self.grid_UNIT + self.grid_origx_center,
                                 self.grid_UNIT),
                random.randrange(self.grid_origy_center, self.grid_columnNum * self.grid_UNIT + self.grid_origy_center,
                                 self.grid_UNIT)]
            if new_loc not in item.AllItemsCurrentPosition_list.values():
                break

        if place:
            self.show(new_loc)

        return new_loc


DISCOUNT = 0.99
MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
BATCH_SIZE = 256  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5


class DQNAgent:

    def __init__(self):
        # 1. To create a new model
        # self.model = self.create_model()

        # 2. To retrain a previously stored model (change the name accordingly
        file_name = 'models/DQNModel_' + str(3) + 'Targets_V6.h5'
        self.model = load_model(file_name)
        self.model.compile(loss="mse", optimizer='Adam', metrics=['accuracy'])

        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.update_counter = 0

    def create_model(self):
        model = Sequential()

        model.add(
            Conv2D(256, (3, 3), input_shape=(8, 8, 1)))  # OBSERVATION_SPACE_VALUES = (10, 10, 3) a 10x10 RGB image.
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(1, 1)))
        model.add(Dropout(0.2))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

        model.add(Dense(64))
        model.add(PReLU())

        model.add(Dense(4, activation='linear'))
        model.compile(loss="mse", optimizer='Adam', metrics=['accuracy'])
        return model

    def update_memory(self, transition):
        self.memory.append(transition)

    def train(self, terminal):
        if len(self.memory) < MIN_MEMORY_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)

        current_state_list = np.array([transition[0] for transition in batch]) / 3
        current_qs_list = self.model.predict(current_state_list)

        next_state_list = np.array([transition[3] for transition in batch]) / 3
        next_qs_list = self.target_model.predict(next_state_list)

        X = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(batch):

            if not done:
                max_future_q = np.max(next_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q

            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state.reshape(8, 8, 1))
            y.append(current_qs)

        self.model.fit(np.array(X) / 3, np.array(y), batch_size=BATCH_SIZE, verbose=0, shuffle=False)
        if terminal:
            self.update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.update_counter = 0

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape) / 3)[0]


root = Tk()
root.title("single agent Q-Learning")
root.attributes("-fullscreen", False)
w, h = root.maxsize()
app = App(root)
root.bind('<Delete>', app.delete_item)
root.mainloop()
