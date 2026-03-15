import json
import random
import time


GRID = [
    [-1, -10, -1, -1, -1, -1, -1, -1, -1, -1, 10],
    [-1, -10, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -10, -10, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -10, -10, -1, -1, -1, -10, -10, -1, -1],
    [-1, -1, -1, -1, -1, -10, -1, -1, -1, -1, -1],
    [-1, -10, -1, -1, -1, -10, -1, -1, -10, -1, -1],
    [-1, -10, -1, -1, -1, -1, -1, -1, -10, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -10, -10, -1, -1, -1, -10, -10],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
]

ROWS = len(GRID)
COLS = len(GRID[0])

START = (5, 9)
GOAL = (0, 10)

ACTIONS = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1),   # right
}

ACTION_NAMES = {
    0: "up",
    1: "down",
    2: "left",
    3: "right",
}

ALPHA = 0.1
GAMMA = 0.99
EPSILON = 0.5
EPISODES = 8000
MAX_STEPS = 400

EXPERIMENT_METADATA = {
    "obstacle_density_case": "case 1",
    "obstacle_density_percent": 15.0,
    "reward_type": {
        "goal_reward": 10,
        "obstacle_penalty": -15,
        "out_of_bounds_penalty": -10,
        "step_penalty": -1,
    },
}


def is_valid_position(x, y):
    return 0 <= x < ROWS and 0 <= y < COLS


def is_obstacle(x, y):
    return GRID[x][y] == -10


def is_goal(x, y):
    return (x, y) == GOAL


def initialize_q_table():
    q_table = {}
    for x in range(ROWS):
        for y in range(COLS):
            q_table[f"{x},{y}"] = [0.0, 0.0, 0.0, 0.0]
    return q_table


def choose_action(q_table, state, epsilon):
    state_key = f"{state[0]},{state[1]}"

    if random.random() < epsilon:
        return random.randint(0, 3)

    q_values = q_table[state_key]
    max_q = max(q_values)

    best_actions = [action for action, value in enumerate(q_values) if value == max_q]
    return random.choice(best_actions)


def step(state, action):
    x, y = state
    dx, dy = ACTIONS[action]
    new_x = x + dx
    new_y = y + dy

    if not is_valid_position(new_x, new_y):
        return state, -10, False

    if is_obstacle(new_x, new_y):
        return state, -15, False

    if is_goal(new_x, new_y):
        return (new_x, new_y), 10, True

    return (new_x, new_y), -1, False


def train_q_learning():
    q_table = initialize_q_table()
    total_steps = 0

    for episode in range(EPISODES):
        state = START

        for _ in range(MAX_STEPS):
            total_steps += 1

            action = choose_action(q_table, state, EPSILON)
            next_state, reward, done = step(state, action)

            state_key = f"{state[0]},{state[1]}"
            next_state_key = f"{next_state[0]},{next_state[1]}"

            old_q = q_table[state_key][action]
            next_max_q = max(q_table[next_state_key])

            q_table[state_key][action] = old_q + ALPHA * (
                reward + GAMMA * next_max_q - old_q
            )

            state = next_state

            if done:
                break

        if (episode + 1) % 500 == 0:
            print(f"Episode {episode + 1}/{EPISODES} completed")

    return q_table, total_steps


def extract_greedy_path(q_table):
    state = START
    path = [state]
    visited = {state}

    for _ in range(MAX_STEPS):
        if state == GOAL:
            break

        state_key = f"{state[0]},{state[1]}"
        q_values = q_table[state_key]
        action = q_values.index(max(q_values))

        next_state, _, done = step(state, action)

        if next_state == state or next_state in visited:
            break

        path.append(next_state)
        visited.add(next_state)
        state = next_state

        if done:
            break

    return path


def save_q_table(q_table, filename="q_table.json"):
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(q_table, file, indent=4)


def save_path(path, filename="path.json"):
    path_data = {
        "trajectory": [{"point": [x, y]} for x, y in path]
    }
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(path_data, file, indent=4)


def save_experiment_results(total_steps, execution_time, filename="experiment_results.json"):
    experiment_data = {
        "obstacle_density_case": EXPERIMENT_METADATA["obstacle_density_case"],
        "obstacle_density_percent": EXPERIMENT_METADATA["obstacle_density_percent"],
        "reward_type": EXPERIMENT_METADATA["reward_type"],
        "total_steps": total_steps,
        "execution_time_seconds": round(execution_time, 4),
        "hyperparameters": {
            "alpha": ALPHA,
            "gamma": GAMMA,
            "epsilon": EPSILON,
            "episodes": EPISODES,
            "max_steps": MAX_STEPS,
        },
    }

    with open(filename, "w", encoding="utf-8") as file:
        json.dump(experiment_data, file, indent=4)


def main():
    start_time = time.perf_counter()

    q_table, total_steps = train_q_learning()
    save_q_table(q_table, "q_table.json")

    best_path = extract_greedy_path(q_table)
    save_path(best_path, "path.json")

    execution_time = time.perf_counter() - start_time
    save_experiment_results(total_steps, execution_time, "experiment_results.json")

    print("Training completed.")
    print("Q-table saved to q_table.json")
    print("Experiment results saved to experiment_results.json")
    print("Learned path:")
    print(best_path)


if __name__ == "__main__":
    main()