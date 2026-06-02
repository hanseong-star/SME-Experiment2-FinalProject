import os

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn




# ============================================================
# [설계 파라미터] 학습된 DQN 모델 경로
# ============================================================
MODEL_PATH = "dqn_model.pt"


# ============================================================
# [설계 파라미터] 맵 설정
# train.py와 반드시 동일해야 함
# ============================================================
X_MIN = -60.0
X_MAX = 60.0
Y_MIN = -30.0
Y_MAX = 30.0

STEP_SIZE = 1.0
MAX_STEPS = 200


ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_STOP = 4
NUM_ACTIONS = 5


#디버깅용
stop_count = 0
max_step_count = 0
step_list = []

# # [설계 파라미터] validation 시작 위치 설정
# # "center" 또는 "random"
# VALIDATION_START_MODE = "random"
#
# # [설계 파라미터] validation random 시작 위치 고정용 seed
# VALIDATION_RANDOM_SEED = 42


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
    return state # 기존
#
# def make_state(x, y, d_hat_u, bs_positions):
#     x_n, y_n = normalize_position(x, y)
#     d_n = normalize_distances(d_hat_u)
#     bs_n = normalize_bs_positions(bs_positions)
#
#     state = np.concatenate(([x_n, y_n], d_n, bs_n)).astype(np.float32)
#     return state


def clamp_position(x, y):
    x = min(max(x, X_MIN), X_MAX)
    y = min(max(y, Y_MIN), Y_MAX)
    return x, y


def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    state_dim = int(checkpoint["state_dim"])
    action_dim = int(checkpoint["action_dim"])

    model = DQN(state_dim, action_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def your_algorithm(d_hat_u, p_bs):
    """
    DQN을 이용해 하나의 사용자 위치를 추정하는 함수.

    입력:
        d_hat_u: shape (18,)
        p_bs   : shape (2, 18), 현재 설계에서는 상태에 넣지 않음

    출력:
        estimated_position: shape (2,)
            [x_hat, y_hat]
    """

    model = load_model()

    # # [설계 파라미터] validation 시작 위치
    # if VALIDATION_START_MODE == "random":
    #     rng = np.random.default_rng(VALIDATION_RANDOM_SEED + user_idx)
    #     x = float(rng.uniform(X_MIN, X_MAX))
    #     y = float(rng.uniform(Y_MIN, Y_MAX))
    # else:
    #     x = (X_MIN + X_MAX) / 2.0
    #     y = (Y_MIN + Y_MAX) / 2.0
    x = (X_MIN + X_MAX) / 2.0
    y = (Y_MIN + Y_MAX) / 2.0 #기본 가운데로 시작

    d_hat_u = np.asarray(d_hat_u, dtype=np.float32).reshape(-1)

    for _ in range(MAX_STEPS):
        state = make_state(x, y, d_hat_u)

        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = model(state_tensor)
            action = int(torch.argmax(q_values, dim=1).item())

        if action == ACTION_STOP:
            break

        if action == ACTION_UP:
            y += STEP_SIZE
        elif action == ACTION_DOWN:
            y -= STEP_SIZE
        elif action == ACTION_LEFT:
            x -= STEP_SIZE
        elif action == ACTION_RIGHT:
            x += STEP_SIZE

        x, y = clamp_position(x, y)

    return np.array([x, y], dtype=float)


def main():
    # 1) 입력 데이터 로드 — 채점기가 같은 폴더에 .mat 파일 자동 배치
    mat_path = "DH_FR1.mat"
    data = sio.loadmat(mat_path, squeeze_me=False)

    BS_positions = np.asarray(data["BS_positions"], dtype=float)
    d_hat = np.asarray(data["d_hat"], dtype=float)
    p = np.asarray(data["p"], dtype=float)

    num_user = d_hat.shape[1]
    p_hat = np.zeros((2, num_user), dtype=float)

    for u in range(num_user):
        p_hat[:, u] = your_algorithm(d_hat[:, u], BS_positions)

    # errors = np.sqrt(np.sum((p_hat - p) ** 2, axis=0))

    # 성능 평가에 사용
    # print("validation file:", MAT_PATH)
    # print("p_hat shape:", p_hat.shape)
    # print("samples:", num_user)
    # print("mean error:", float(np.mean(errors)))
    # print("median error:", float(np.median(errors)))
    # print("rmse:", float(np.sqrt(np.mean(errors ** 2))))
    # print("90% error:", float(np.percentile(errors, 90)))
    # print("max error:", float(np.max(errors)))

    return p_hat


if __name__ == "__main__":
    main()
