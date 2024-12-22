import random
import numpy as np
import time

# Constants
EMPTY = 0
PLAYER_AGENT_1 = 1
PLAYER_AGENT_2 = 2
TAKEN = -1
DRAW = 3

BOARD_SIZE = 8


def printboard(state, agent1_pos, agent2_pos):
    for i in range(BOARD_SIZE):
        row = []
        for j in range(BOARD_SIZE):
            if (i, j) == agent1_pos:
                row.append("A")  # Agent 1 marker
            elif (i, j) == agent2_pos:
                row.append("B")  # Agent 2 marker
            elif state[i][j] == TAKEN:
                row.append("-")  # Taken tile
            else:
                row.append(".")  # Empty tile
        print(' '.join(row))
    print()


def emptystate():
    """Create an empty board."""
    return [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]


def gameover(state, agent1_pos, agent2_pos):
    if state[agent1_pos[0]][agent1_pos[1]] == TAKEN:
        print("Agent 1 lost by revisiting a tile!")
        return True
    if state[agent2_pos[0]][agent2_pos[1]] == TAKEN:
        print("Agent 2 lost by revisiting a tile!")
        return True
    if agent1_pos == agent2_pos:
        print("Agents collided! It's a draw!")
        return True
    return False


def move(position, direction):
    moves = {
        'up': (-1, 0),
        'down': (1, 0),
        'left': (0, -1),
        'right': (0, 1)
    }
    delta = moves.get(direction)
    if delta:
        return position[0] + delta[0], position[1] + delta[1]
    return position


def is_valid_move(position):
    return 0 <= position[0] < BOARD_SIZE and 0 <= position[1] < BOARD_SIZE


class Agent:
    def __init__(self, player):
        self.player = player
        self.q_table = np.zeros((BOARD_SIZE * BOARD_SIZE, 4))  # 4 actions: up, down, left, right
        self.learning_rate = 0.8
        self.discount_factor = 0.95
        self.epsilon = 0.1

    def encode_state(self, position):
        """Encode the (row, col) position into a single integer."""
        return position[0] * BOARD_SIZE + position[1]

    def train(self, episodes=10000):
        for _ in range(episodes):
            position = (0, 0)  # Agent starts at the top-left corner
            visited = set()
            done = False

            while not done:
                state_idx = self.encode_state(position)

                # Choose action (exploration or exploitation)
                if random.uniform(0, 1) < self.epsilon:
                    action = random.choice(range(4))  # Random action: 0=up, 1=down, 2=left, 3=right
                else:
                    action = np.argmax(self.q_table[state_idx])

                # Determine the next position based on the action
                direction = ['up', 'down', 'left', 'right'][action]
                next_position = move(position, direction)

                if not is_valid_move(next_position) or next_position in visited:
                    reward = -10  # Heavier penalty for invalid or revisited moves
                    done = True
                else:
                    reward = 1  # Reward valid moves
                    visited.add(next_position)
                    position = next_position

                if is_valid_move(next_position):
                    next_state_idx = self.encode_state(next_position)
                    next_max = np.max(self.q_table[next_state_idx])
                else:
                    next_max = 0  # No future reward for invalid moves

                # Update Q-Table
                self.q_table[state_idx, action] += self.learning_rate * (
                    reward + self.discount_factor * next_max - self.q_table[state_idx, action]
                )

                if done:
                    break

    def action(self, state, position):
        state_idx = self.encode_state(position)
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(range(4))  # Exploration
        else:
            action = np.argmax(self.q_table[state_idx])  # Exploitation
        direction = ['up', 'down', 'left', 'right'][action]
        new_pos = move(position, direction)
        return new_pos


def play(agent1, agent2):
    state = emptystate()
    agent1_pos = (0, 0)
    agent2_pos = (BOARD_SIZE - 1, BOARD_SIZE - 1)
    state[agent1_pos[0]][agent1_pos[1]] = PLAYER_AGENT_1
    state[agent2_pos[0]][agent2_pos[1]] = PLAYER_AGENT_2

    while True:
        printboard(state, agent1_pos, agent2_pos)  # Display current board
        time.sleep(1)  # Add 1-second timeout per round

        # Agent 1's turn
        new_agent1_pos = agent1.action(state, agent1_pos)
        state[agent1_pos[0]][agent1_pos[1]] = TAKEN  # Mark tile as taken
        agent1_pos = new_agent1_pos
        if gameover(state, agent1_pos, agent2_pos):
            break

        # Agent 2's turn
        new_agent2_pos = agent2.action(state, agent2_pos)
        state[agent2_pos[0]][agent2_pos[1]] = TAKEN  # Mark tile as taken
        agent2_pos = new_agent2_pos
        if gameover(state, agent1_pos, agent2_pos):
            break


if __name__ == "__main__":
    agent1 = Agent(PLAYER_AGENT_1)
    agent2 = Agent(PLAYER_AGENT_2)
    agent1.train(episodes=10000)  # Improve training with more episodes
    agent2.train(episodes=10000)
    play(agent1, agent2)
