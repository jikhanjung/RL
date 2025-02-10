import numpy as np
import random

# -----------------------------
# 1) 데이터 및 기본 함수 정의
# -----------------------------
def load_morphological_data(filepath):
    """
    사용 예시:
      - 각 택사(taxon)에 대해 여러 이산형 문자(0,1,2 등)를 읽어온다.
      - NxM 매트릭스로 반환 (N=택사 수, M=특징 수)
    """
    # 여기서는 예시로 간단히 무작위 데이터 생성
    # 실제 사용 시 파일에서 읽어오는 로직 구현
    num_taxa = 5
    num_chars = 4
    # 0~2 범주의 문자상태를 랜덤으로 생성
    data_matrix = np.random.randint(0, 3, size=(num_taxa, num_chars))
    taxa_names = [f"Taxon_{i}" for i in range(num_taxa)]
    return data_matrix, taxa_names

def fitch_parsimony_score(tree, data_matrix):
    """
    트리에 대한 패싱모니 점수를 Fitch 알고리즘으로 계산.
    - tree: 트리 구조(부모-자식 관계, 노드 정보 등) 
    - data_matrix: 각 택사별 문자행렬 (N x M)
    - 간단히, 모든 내부 노드를 후위순회하며
      각 문자별 변화 횟수를 세어 합산.
    """
    # 여기서는 pseudo-code 형식으로만 제시
    # 실제로는 tree 노드별 상태 집합(set)을 bitset 등으로 관리해야 효율적
    total_changes = 0
    # 예시: 모든 노드에 대해 각 문자의 최소 변화 횟수를 합산
    # 내부적으로 postorder traversal 필요
    # ...
    return total_changes


# --------------------------------
# 2) 트리 표현 및 연산(액션) 정의
# --------------------------------
class PhyloTree:
    """
    간단한 계통수 자료구조 예시.
    실제 구현 시:
     - 노드: 각 택사(또는 내부 노드)
     - edges: 부모-자식 연결
     - 루트, 리프 등 속성
    """
    def __init__(self, taxa_indices):
        # 예시: 리프만 있는 단순 star-tree 형태로 초기화
        self.taxa_indices = taxa_indices  # [0, 1, 2, 3, ...]
        # 실제로는 노드/간선 정보를 저장해야 함
        self.edges = []
        # 간단히 모든 리프를 root에 직결했다고 가정
        # ...

    def clone(self):
        # 트리 복제(깊은 복사)
        new_tree = PhyloTree(self.taxa_indices.copy())
        # edges 등도 복사
        new_tree.edges = [e for e in self.edges]
        return new_tree

    def nni_move(self):
        """
        간단히 트리에서 임의의 내부 가지 선택 후 NNI 적용
        실제로는 가능한 NNI 후보 리스트 중 하나를 샘플링
        """
        # pseudo-code
        # 1. 내부 가지 선택
        # 2. 해당 edge 주변 서브트리 스왑
        pass

    def spr_move(self):
        """
        임의의 서브트리를 잘라 다른 가지에 붙이는 SPR
        """
        pass

    def tbr_move(self):
        """
        TBR: 트리를 두 부분으로 절단한 뒤 재연결
        """
        pass

    def get_neighbors(self, move_type='NNI'):
        """
        현재 트리 상태에서 가능한 모든 인접 트리(이웃) 생성
        move_type에 따라 NNI/SPR/TBR 등 달라질 수 있음
        """
        neighbors = []
        # 예시) NNI만 구현
        # 실제론 모든 내부 edge에 대해 2가지 NNI
        return neighbors


# --------------------------
# 3) 환경(Env) 클래스 설계
# --------------------------
class ParsimonyEnv:
    """
    OpenAI Gym 스타일을 흉내낸 간단한 환경 예시:
    - state: 현재 트리
    - action: 특정 branch-swap 연산 (NNI, SPR, TBR 등)
    - reward: 점수 개선(패싱모니 감소)에 대한 보상
    """
    def __init__(self, data_matrix):
        self.data_matrix = data_matrix
        self.num_taxa = data_matrix.shape[0]
        # 초기 트리: 간단히 모든 taxon을 root에 직결한 star-tree 
        init_taxa = list(range(self.num_taxa))
        self.tree = PhyloTree(init_taxa)
        self.current_score = fitch_parsimony_score(self.tree, self.data_matrix)

    def reset(self):
        # 환경 재설정: 트리 초기화
        init_taxa = list(range(self.num_taxa))
        self.tree = PhyloTree(init_taxa)
        self.current_score = fitch_parsimony_score(self.tree, self.data_matrix)
        return self._get_observation()

    def _get_observation(self):
        """
        트리 구조를 상태로 어떻게 표현할지는 난제.
        간단히 '현재 패싱모니 점수, 노드 연결정보' 등을 숫자/벡터로 변환하여 반환
        (RL NN에 넣을 feature로 사용)
        """
        # 여기서는 예시로 점수와 리프 수만 반환
        obs = (self.current_score, len(self.tree.taxa_indices))
        return obs

    def step(self, action):
        """
        action에 따라 트리를 수정하고,
        새로운 score 계산 -> 보상 계산
        """
        new_tree = self.tree.clone()

        # action을 어떻게 정의하느냐에 따라 달라짐
        if action == 0:
            # 예) NNI move
            new_tree.nni_move()
        elif action == 1:
            # SPR
            new_tree.spr_move()
        elif action == 2:
            # TBR
            new_tree.tbr_move()
        # 더 세분화된 액션 (어느 edge에 적용할지)도 가능

        new_score = fitch_parsimony_score(new_tree, self.data_matrix)

        # 보상 설계: 점수가 낮아지면 + 보상, 높아지면 - 보상
        reward = float(self.current_score - new_score)
        # state 갱신
        self.tree = new_tree
        self.current_score = new_score

        # done 조건 (간단 예시): 
        # 일정 횟수 이상 step 했거나, 스코어가 0이 되면 종료
        done = (new_score == 0)

        obs = self._get_observation()
        return obs, reward, done, {}

# -----------------------------
# 4) 간단한 DQN 에이전트 예시
# -----------------------------
import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=1.0):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.action_dim = action_dim

        self.qnet = QNetwork(state_dim, action_dim)
        self.target_qnet = QNetwork(state_dim, action_dim)
        self.target_qnet.load_state_dict(self.qnet.state_dict())

        self.optimizer = optim.Adam(self.qnet.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.memory = []
        self.batch_size = 32

    def get_action(self, state):
        # eps-greedy
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            state_t = torch.FloatTensor([state])
            q_values = self.qnet(state_t)
            action = torch.argmax(q_values, dim=1).item()
            return action

    def remember(self, transition):
        self.memory.append(transition)
        if len(self.memory) > 10000:
            self.memory.pop(0)

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.FloatTensor(states)
        actions_t = torch.LongTensor(actions).unsqueeze(1)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1)
        next_states_t = torch.FloatTensor(next_states)
        dones_t = torch.BoolTensor(dones).unsqueeze(1)

        # Q(s,a)
        q_values = self.qnet(states_t).gather(1, actions_t)

        # max Q'(s', a')
        with torch.no_grad():
            next_q_values = self.target_qnet(next_states_t).max(dim=1, keepdim=True)[0]
            target = rewards_t + self.gamma * next_q_values * (~dones_t)

        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_qnet.load_state_dict(self.qnet.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# -----------------------------
# 5) 메인 루프: 학습 실행
# -----------------------------
def run_training(num_episodes=50, max_steps=100):
    data_matrix, taxa_names = load_morphological_data("dummy_path")
    env = ParsimonyEnv(data_matrix)

    # (current_score, num_taxa) -> 예시로 state_dim=2, action_dim=3(NNI,SPR,TBR)
    state_dim = 2
    action_dim = 3
    agent = DQNAgent(state_dim, action_dim)

    update_target_interval = 10

    for ep in range(num_episodes):
        state = env.reset()  # (score, num_taxa)
        episode_reward = 0
        for step in range(max_steps):
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)

            agent.remember((state, action, reward, next_state, done))
            agent.train_step()

            state = next_state
            episode_reward += reward

            if done:
                print(f"[Episode {ep}] Step={step}, reward={episode_reward}, final_score={env.current_score}")
                break

        # 타겟 네트워크 업데이트
        if ep % update_target_interval == 0:
            agent.update_target()

        agent.decay_epsilon()


if __name__ == "__main__":
    run_training()
