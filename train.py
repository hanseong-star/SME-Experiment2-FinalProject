import os
import random
from collections import deque

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# [설계 파라미터] 파일 경로
# ============================================================
TRAIN_MAT_PATH = "../DataSet/train_500.mat"
MODEL_SAVE_PATH = "dqn_model.pt"


# ============================================================
# [설계 파라미터] 맵 설정
# 실제 좌표 범위: x = -60 ~ 60, y = -30 ~ 30
# ============================================================
X_MIN = -60.0
X_MAX = 60.0
Y_MIN = -30.0
Y_MAX = 30.0

STEP_SIZE = 1.0
MAX_STEPS = 200


# ============================================================
# [설계 파라미터] DQN 학습 설정
# ============================================================
EPISODES = 3000
BATCH_SIZE = 64
MEMORY_SIZE = 50000

GAMMA = 0.98
LEARNING_RATE = 1e-3

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995

TARGET_UPDATE_INTERVAL = 100


# ============================================================
# [설계 파라미터] 보상 설정
# ============================================================
# MAX_STEP_FAIL_REWARD = -80.0

STOP_SUCCESS_RADIUS = 2.0
STOP_MID_RADIUS = 5.0

STOP_SUCCESS_REWARD = 20.0
STOP_MID_REWARD = 5.0
STOP_FAIL_REWARD = -20.0

OUT_OF_MAP_REWARD = -5.0

# [설계 파라미터] 이동 보상
MOVE_TIME_PENALTY = 0.01
DISTANCE_REWARD_SCALE = 1.0

# [설계 파라미터] Smooth Step STOP reward
STOP_BASE_REWARD = -20.0

STOP_SUCCESS_RADIUS = 2.0
STOP_MID_RADIUS = 5.0
STOP_FAR_RADIUS = 12.0

STOP_MID_BONUS = 25.0
STOP_SUCCESS_BONUS = 15.0
STOP_FAR_PENALTY = 20.0

STOP_MID_TAU = 0.35
STOP_SUCCESS_TAU = 0.25
STOP_FAR_TAU = 4.0


# # [설계 파라미터] STOP 부드러운 곡선 보상
# STOP_REWARD_SCALE = 40.0
# STOP_SIGMA = 3.0
# STOP_DISTANCE_PENALTY = 2.0
# STOP_REWARD_MIN = -100.0
# STOP_REWARD_MAX = 40.0
#
# STOP_REWARD_SCALE = 30.0
# STOP_TARGET_RADIUS = 5.0
# STOP_SMOOTHNESS = 3.0

# ============================================================
# [설계 파라미터] 시작 위치 설정
# "center" 또는 "random"
# ============================================================
START_MODE = "random"


ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_STOP = 4
NUM_ACTIONS = 5


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize_position(x, y):
    x_n = (x - X_MIN) / (X_MAX - X_MIN)
    y_n = (y - Y_MIN) / (Y_MAX - Y_MIN)
    return x_n, y_n

# def normalize_bs_positions(bs_positions):
#     bs_positions = np.asarray(bs_positions, dtype=np.float32)
#
#     bs_x = bs_positions[0, :]
#     bs_y = bs_positions[1, :]
#
#     bs_x_n = (bs_x - X_MIN) / (X_MAX - X_MIN)
#     bs_y_n = (bs_y - Y_MIN) / (Y_MAX - Y_MIN)
#
#     return np.concatenate([bs_x_n, bs_y_n]).astype(np.float32)


def normalize_distances(d):
    d = np.asarray(d, dtype=np.float32)
    return d / 320.0


def make_state(x, y, d_hat_u):
    x_n, y_n = normalize_position(x, y)
    d_n = normalize_distances(d_hat_u)
    state = np.concatenate(([x_n, y_n], d_n)).astype(np.float32)
    return state #기존

# def make_state(x, y, d_hat_u, bs_positions):
#     x_n, y_n = normalize_position(x, y)
#     d_n = normalize_distances(d_hat_u)
#     bs_n = normalize_bs_positions(bs_positions)
# 
#     state = np.concatenate(([x_n, y_n], d_n, bs_n)).astype(np.float32)
#     return state  #bs position 포함 버전


def clamp_position(x, y):
    x = min(max(x, X_MIN), X_MAX)
    y = min(max(y, Y_MIN), Y_MAX)
    return x, y


def euclidean_error(x, y, target):
    dx = x - float(target[0])
    dy = y - float(target[1])
    return float(np.sqrt(dx * dx + dy * dy))


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        states = np.array([b[0] for b in batch], dtype=np.float32)
        actions = np.array([b[1] for b in batch], dtype=np.int64)
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = np.array([b[3] for b in batch], dtype=np.float32)
        dones = np.array([b[4] for b in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class PositionEnv:
    def __init__(self, d_hat, p):
        self.d_hat = np.asarray(d_hat, dtype=np.float32)
        self.p = np.asarray(p, dtype=np.float32)
        # self.BS_positions = np.asarray(BS_positions, dtype=np.float32)
        self.num_user = self.d_hat.shape[1]

        self.user_idx = None
        self.target = None
        self.d_hat_u = None
        self.x = None
        self.y = None
        self.step_count = None
        self.prev_error = None

    def reset(self):
        self.user_idx = np.random.randint(0, self.num_user)
        self.target = self.p[:, self.user_idx]
        self.d_hat_u = self.d_hat[:, self.user_idx]

        if START_MODE == "center":
            self.x = (X_MIN + X_MAX) / 2.0
            self.y = (Y_MIN + Y_MAX) / 2.0
        else:
            self.x = np.random.uniform(X_MIN, X_MAX)
            self.y = np.random.uniform(Y_MIN, Y_MAX)

        self.step_count = 0
        self.prev_error = euclidean_error(self.x, self.y, self.target)

        return make_state(self.x, self.y, self.d_hat_u)

    def step(self, action):
        done = False
        reward = 0.0

        if action == ACTION_STOP:
            curr_error = euclidean_error(self.x, self.y, self.target)



            if curr_error <= STOP_SUCCESS_RADIUS:
                reward = STOP_SUCCESS_REWARD
            elif curr_error <= STOP_MID_RADIUS:
                reward = STOP_MID_REWARD
            else:
                reward = STOP_FAIL_REWARD  # 기존

            # reward = STOP_REWARD_SCALE * np.tanh(
            #     (STOP_TARGET_RADIUS - curr_error) / STOP_SMOOTHNESS
            # )

            # mid_term = STOP_MID_BONUS * sigmoid(
            #     (STOP_MID_RADIUS - curr_error) / STOP_MID_TAU
            # )
            #
            # success_term = STOP_SUCCESS_BONUS * sigmoid(
            #     (STOP_SUCCESS_RADIUS - curr_error) / STOP_SUCCESS_TAU
            # )
            #
            # reward = STOP_BASE_REWARD + success_term + mid_term #- far_term # Smooth Step STOP Reward

            done = True
            next_state = make_state(self.x, self.y, self.d_hat_u)
            return next_state, reward, done

        old_x = self.x
        old_y = self.y

        if action == ACTION_UP:
            self.y += STEP_SIZE
        elif action == ACTION_DOWN:
            self.y -= STEP_SIZE
        elif action == ACTION_LEFT:
            self.x -= STEP_SIZE
        elif action == ACTION_RIGHT:
            self.x += STEP_SIZE

        out_of_map = (
            self.x < X_MIN or self.x > X_MAX or
            self.y < Y_MIN or self.y > Y_MAX
        )

        self.x, self.y = clamp_position(self.x, self.y)

        curr_error = euclidean_error(self.x, self.y, self.target)

        reward = DISTANCE_REWARD_SCALE * (self.prev_error - curr_error)
        reward -= MOVE_TIME_PENALTY  #기존

        # delta_error = self.prev_error - curr_error
        # delta_error = np.clip(delta_error, -1.0, 1.0)
        #
        # reward = DISTANCE_REWARD_SCALE * delta_error
        # reward -= MOVE_TIME_PENALTY #이동 보상 clip 추가

        if out_of_map:
            reward += OUT_OF_MAP_REWARD
            self.x = old_x
            self.y = old_y
            curr_error = self.prev_error

        self.prev_error = curr_error
        self.step_count += 1

        if self.step_count >= MAX_STEPS:
            done = True

        next_state = make_state(self.x, self.y, self.d_hat_u)
        return next_state, float(reward), done


def train_dqn():
    set_seed(42)

    data = sio.loadmat(TRAIN_MAT_PATH, squeeze_me=False)
    d_hat = np.asarray(data["d_hat"], dtype=np.float32)
    p = np.asarray(data["p"], dtype=np.float32)
    # BS_positions = np.asarray(data["BS_positions"], dtype=np.float32)

    env = PositionEnv(d_hat, p)

    state_dim = 2 + d_hat.shape[0] #기존
    # state_dim = 2 + d_hat.shape[0] + 2 * BS_positions.shape[1]
    action_dim = NUM_ACTIONS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayBuffer(MEMORY_SIZE)

    epsilon = EPSILON_START
    total_step = 0

    for episode in range(1, EPISODES + 1):
        state = env.reset()
        episode_reward = 0.0

        for _ in range(MAX_STEPS):
            total_step += 1

            if random.random() < epsilon:
                action = random.randrange(action_dim)
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    q_values = policy_net(state_tensor)
                    action = int(torch.argmax(q_values, dim=1).item())

            next_state, reward, done = env.step(action)
            memory.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            if len(memory) >= BATCH_SIZE:
                states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

                states = torch.tensor(states, dtype=torch.float32).to(device)
                actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device)
                rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
                next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
                dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

                q_values = policy_net(states).gather(1, actions)

                with torch.no_grad():
                    next_q_values = target_net(next_states).max(1, keepdim=True)[0]
                    target_q_values = rewards + GAMMA * next_q_values * (1.0 - dones) #기존 ＤＱＮ

                #
                # double DQN
                # with torch.no_grad():
                #     next_actions = policy_net(next_states).argmax(1, keepdim=True)
                #     next_q_values = target_net(next_states).gather(1, next_actions)
                #     target_q_values = rewards + GAMMA * next_q_values * (1.0 - dones)


                loss = nn.MSELoss()(q_values, target_q_values) #MSELoss
                # loss = nn.SmoothL1Loss()(q_values, target_q_values) #SmoothL1Loss

                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0) #gradient clipping 추가
                optimizer.step()

            if total_step % TARGET_UPDATE_INTERVAL == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        if episode % 100 == 0:
            print(
                f"Episode {episode:5d} | "
                f"Reward {episode_reward:8.3f} | "
                f"Epsilon {epsilon:.3f}"
            )

    save_data = {
        "model_state_dict": policy_net.state_dict(),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "x_min": X_MIN,
        "x_max": X_MAX,
        "y_min": Y_MIN,
        "y_max": Y_MAX,
        "step_size": STEP_SIZE,
    }

    torch.save(save_data, MODEL_SAVE_PATH)
    print(f"Saved model to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train_dqn()