import numpy as np

import gymnasium as gym
from gymnasium import spaces

import matplotlib.pyplot as plt

min_int8 = np.iinfo(np.int8).min


def _apply_kernel(board: np.ndarray, kernel: np.ndarray, row: int, col: int):
    """Applies the kernel to the board at a specific location accounting for padding and with centering on the actual placement"""
    ker_row_sub_l: int = 0
    ker_col_sub_l: int = 0
    ker_row_sub_r: int = 0
    ker_col_sub_r: int = 0
    board_row_sub_l: int = -1
    board_col_sub_l: int = -1
    board_row_sub_r: int = -1
    board_col_sub_r: int = -1
    if row == 0:
        board_row_sub_l = 0
        ker_row_sub_l = 1
    elif row == board.shape[0] - 1:
        board_row_sub_r = 0
        ker_row_sub_r = -1
    if col == 0:
        board_col_sub_l = 0
        ker_col_sub_l = 1
    elif col == board.shape[1] - 1:
        board_col_sub_r = 0
        ker_col_sub_r = -1

    return (
        board[
            board_row_sub_l + row : board_row_sub_r + row + kernel.shape[0],
            board_col_sub_l + col : board_col_sub_r + col + kernel.shape[1],
        ]
        * kernel[
            ker_row_sub_l : ker_row_sub_r + kernel.shape[0],
            ker_col_sub_l : ker_col_sub_r + kernel.shape[1],
        ]
    )


def zero_corners(arr: np.ndarray):
    """Sets the corners of the array to zero"""
    arr[0, 0] = 0
    arr[0, -1] = 0
    arr[-1, 0] = 0
    arr[-1, -1] = 0
    return arr


class BattleshipEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple Battleship game environment.
    The player has to guess the location of the opponent's ship.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        board_size: tuple[int, int] = (10, 10),
        ship_sizes: list[int] = [5, 4, 3, 3, 2],
        episode_steps=100,
    ):
        super(BattleshipEnv, self).__init__()

        # Game Constants
        self.board_size = board_size
        self.ship_sizes = ship_sizes
        self.total_shots = sum(ship_sizes)
        self.ship_count = len(ship_sizes)
        self.ship_kernels_horiz = [
            zero_corners(np.ones((3, ship_size), dtype=np.int8))
            for ship_size in ship_sizes
        ]
        self.ship_kernels_vert = [
            zero_corners(np.ones((ship_size, 3), dtype=np.int8))
            for ship_size in ship_sizes
        ]

        # Empty Initializations
        self.board = np.empty((board_size[0], board_size[1]), dtype=np.int8)
        self.og_board = np.empty((2, board_size[0], board_size[1]), dtype=np.int8)
        self.observations = np.empty(
            (2, board_size[0], board_size[1]), dtype=np.bool_
        )  # first layer is for missed shots, second for hits

        # Set maximum episode steps
        self.episode_steps = episode_steps

        # Define action and observation space
        self.action_space = spaces.Discrete(board_size[0] * board_size[1])
        self.observation_space = spaces.MultiBinary(
            (board_size[0], board_size[1], len(ship_sizes) + 1)
        )

        self.reset()

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.done, self.truncated = False, False
        self.step_count = 0

        # Reset the board and observations
        self.board.fill(0)
        self.observations.fill(False)
        self.hits = 0

        # Place ships randomly on the board
        for ship_num, ship_size in enumerate(self.ship_sizes):
            placed = False
            while not placed:
                orientation_vert = self.np_random.choice((True, False))
                if orientation_vert:
                    row = self.np_random.integers(0, self.board_size[0])
                    col = self.np_random.integers(0, self.board_size[1])
                    if row > self.board_size[0] - ship_size:
                        continue
                    kernel = self.ship_kernels_vert[ship_num]
                    if np.all(_apply_kernel(self.board, kernel, row, col) == 0):
                        self.board[row : row + ship_size, col] = ship_num + 1
                        placed = True
                else:
                    row = self.np_random.integers(0, self.board_size[0])
                    col = self.np_random.integers(0, self.board_size[1])
                    if col > self.board_size[1] - ship_size:
                        continue
                    kernel = self.ship_kernels_horiz[ship_num]
                    if np.all(_apply_kernel(self.board, kernel, row, col) == 0):
                        self.board[row, col : col + ship_size] = ship_num + 1
                        placed = True

        np.copyto(self.og_board, self.board)

        return self.observations

    def step(self, action: int):
        if self.done:
            raise RuntimeError("Episode is done. Please reset the environment.")

        reward = -0.1

        row = action // self.board_size[1]
        col = action % self.board_size[1]

        self.step_count += 1

        if self.board[row, col] == 0 or self.board[row, col] == min_int8:
            self.observations[0, row, col] = True
            self.board[row, col] = min_int8
            reward = 0
        elif self.board[row, col] > 0:
            ship_num = self.board[row, col]
            self.observations[1, row, col] = True
            self.board[row, col] = -ship_num
            self.hits += 1
            reward = 1
        else:
            pass  # already hit

        self.done = self.hits == self.total_shots
        self.truncated = self.step_count >= self.episode_steps

        return (self.observations, reward, self.done, self.truncated, {})

    def render(self, mode="human"):
        if mode == "human":
            print("Current Board:")
            print(self.board)
            print("Observations:")
            print(self.observations)
            print("Hits:", self.hits)
            print("Step Count:", self.step_count)
        elif mode == "playing":
            print("Observations:")
            print(self.observations)
        elif mode == "2d":
            plt.imshow(
                self.observations[0] + 2 * self.observations[1],
                cmap="hot",
                interpolation="nearest",
            )
            plt.show()

if __name__ == "__main__":
    env = BattleshipEnv()
    env.reset()

    master = np.zeros((10, 10), dtype=np.int32)
    for _ in range(500000):
        master += env.board >= 1
        env.reset()
    print(master)

    plt.imshow(master, cmap="hot", interpolation="nearest")
    print(np.isclose(master, master.T, atol=2e2))
    plt.show()
    plt.imshow(
        np.isclose(master, master.T, atol=2e2), cmap="hot", interpolation="nearest"
    )
    plt.show()
    plt.imshow(
        np.isclose(master, master.T, atol=6e2), cmap="hot", interpolation="nearest"
    )
    plt.show()
    plt.imshow(np.abs(master - master.T), cmap="hot", interpolation="nearest")
    plt.show()
    plt.imshow(
        np.abs(master - master.T) / (master + master.T),
        cmap="hot",
        interpolation="nearest",
    )
    plt.show()
