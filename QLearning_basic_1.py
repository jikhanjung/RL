import numpy as np
import random


class GridWorldEnv:
    """
    5x5 그리드 환경.
    S: 시작 위치
    G: 목표 위치 (도착 시 +10 보상, 에피소드 종료)
    X: 장애물 (이동 불가, 이동 시도 시 -1 보상, 제자리 유지)
    .: 빈 칸 (이동 가능, 이동 시 -0.1 보상)
    """

    def __init__(self):
        # 격자 정의 (5x5)
        # (행, 열) 기준으로 행이 위->아래, 열이 왼->오른쪽
        self.grid = [
            ['S', '.', '.', 'X', '.'],
            ['.', 'X', '.', 'X', '.'],
            ['.', '.', '.', '.', '.'],
            ['.', 'X', 'X', '.', '.'],
            ['.', '.', '.', '.', 'G']
        ]
        
        # Grid 크기
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])
        
        # 시작 상태 (row, col)
        self.start_state = (0, 0)
        # 현재 상태
        self.state = self.start_state
        
        # 목표 상태 위치 찾기
        self.goal_state = None
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == 'G':
                    self.goal_state = (r, c)
                    break
        
        # 이동 행동 정의 (상, 우, 하, 좌)
        # action 0: 상, 1: 우, 2: 하, 3: 좌
        self.actions = {
            0: (-1, 0),  # 상
            1: (0, +1),  # 우
            2: (+1, 0),  # 하
            3: (0, -1)   # 좌
        }
    
    def reset(self):
        """에이전트를 시작 위치로 되돌리고, 초기 상태를 반환."""
        self.state = self.start_state
        return self.state
    
    def step(self, action):
        """
        action: 0(상), 1(우), 2(하), 3(좌) 중 하나.
        반환:
        - next_state: (row, col)
        - reward
        - done: 목표에 도달 시 True
        - info: 기타 정보(여기서는 빈 dict)
        """
        r, c = self.state
        dr, dc = self.actions[action]
        nr, nc = r + dr, c + dc  # 다음 위치 후보
        
        # 격자 범위를 벗어나는지 확인
        if not (0 <= nr < self.rows and 0 <= nc < self.cols):
            # 벗어났으므로 이동 실패, 보상 -1, 제자리에 유지
            reward = -1
            next_state = (r, c)
            done = False
            return next_state, reward, done, {}
        
        # 장애물인지 확인
        if self.grid[nr][nc] == 'X':
            # 장애물이므로 이동 실패, 보상 -1, 제자리에 유지
            reward = -1
            next_state = (r, c)
            done = False
            return next_state, reward, done, {}
        
        # 목표 지점인지 확인
        if self.grid[nr][nc] == 'G':
            # 목표 도달, 보상 +10, 에피소드 종료
            reward = 10
            next_state = (nr, nc)
            done = True
            self.state = next_state
            return next_state, reward, done, {}
        
        # 그 외(빈 칸) 이동 성공
        reward = -0.1  # 이동 비용
        next_state = (nr, nc)
        done = False
        self.state = next_state
        return next_state, reward, done, {}
    

def q_learning_train(env, num_episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1):
    """
    Q-learning으로 GridWorld 학습.
    - env: GridWorldEnv
    - num_episodes: 학습 에피소드 수
    - alpha: 학습률
    - gamma: 할인율
    - epsilon: 탐험(ε-greedy) 확률
    """
    # 상태: (row, col)
    # 가능한 최대 상태 개수: 5 * 5 = 25
    # 하지만 편의를 위해 (row, col)에 직접 index 매핑을 해도 되고,
    # Q를 (rows, cols, action_size) 형태로 구성해도 된다.
    
    rows, cols = env.rows, env.cols
    action_size = len(env.actions)  # 4가지 (상,우,하,좌)
    
    # Q 테이블을 (rows, cols, action_size) 형태로 0으로 초기화
    Q = np.zeros((rows, cols, action_size))
    
    for episode in range(num_episodes):
        state = env.reset()        # 에피소드 시작 상태 (보통 S)
        done = False
        
        while not done:
            r, c = state
            
            # ε-탐욕(epsilon-greedy) 정책으로 행동 선택
            if random.random() < epsilon:
                action = random.randint(0, action_size - 1)
            else:
                action = np.argmax(Q[r, c, :])
            
            # 환경에서 한 스텝 진행
            next_state, reward, done, _ = env.step(action)
            nr, nc = next_state
            
            # Q(s,a) 업데이트
            old_value = Q[r, c, action]
            next_max = np.max(Q[nr, nc, :])
            
            Q[r, c, action] = old_value + alpha * (reward + gamma * next_max - old_value)
            
            state = next_state
        print_optimal_policy(Q, env)
    return Q


def print_optimal_policy(Q, env):
    """
    학습된 Q 테이블을 바탕으로 각 칸에서의 최적 정책을 표시해본다.
    - Q: 학습된 Q 테이블 (shape = [rows, cols, 4])
    - env: GridWorldEnv
    """
    rows, cols = env.rows, env.cols
    actions = {0: '↑', 1: '→', 2: '↓', 3: '←'}
    
    policy_grid = []
    for r in range(rows):
        row_policy = []
        for c in range(cols):
            # 장애물이거나 목표라면 표시만 하고 건너뛴다
            if env.grid[r][c] == 'X':
                row_policy.append('X')
            elif env.grid[r][c] == 'G':
                row_policy.append('G')
            elif env.grid[r][c] == 'S':
                # S도 최적 행동 표시
                best_action = np.argmax(Q[r, c, :])
                row_policy.append('S(' + actions[best_action] + ')')
            else:
                best_action = np.argmax(Q[r, c, :])
                row_policy.append(actions[best_action])
        policy_grid.append(row_policy)
    
    print("== 학습된 최적 정책 (policy) ==")
    for row in policy_grid:
        print(row)


def test_agent(Q, env, max_steps=20):
    """
    학습된 Q 테이블로 에이전트를 실제로 움직여본다.
    - Q: 학습된 Q 테이블
    - env: GridWorldEnv
    - max_steps: 최대 스텝(루프) 제한
    """
    state = env.reset()
    for step in range(max_steps):
        r, c = state
        action = np.argmax(Q[r, c, :])
        next_state, reward, done, _ = env.step(action)
        
        print(f"Step {step}: 상태={state}, 행동={action}, 보상={reward}, 다음 상태={next_state}")
        state = next_state
        
        if done:
            print("목표에 도달했습니다!")
            return
    print("목표에 도달하지 못했습니다 (max_steps 초과)")


def main():
    env = GridWorldEnv()
    Q = q_learning_train(env, num_episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1)
    
    print("학습 완료!")
    
    # 학습된 최적 정책(방향) 출력
    print_optimal_policy(Q, env)
    
    # 학습 결과를 가지고 실제로 테스트
    print("\n== 에이전트 테스트 ==")
    test_agent(Q, env, max_steps=20)


if __name__ == "__main__":
    main()
