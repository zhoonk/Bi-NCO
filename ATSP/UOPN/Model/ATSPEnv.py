
from dataclasses import dataclass
import torch

from ATSProblemDef import get_random_problems
from ATSProblemDefTest import get_random_problems as get_test_problems


@dataclass
class Reset_State:
    problems: torch.Tensor
    # shape: (batch, problem, 2)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor
    SAMPLE_IDX: torch.Tensor
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, node)


class ATSPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.node_cnt = env_params['node_cnt']
        self.trajectory_size = env_params['trajectory_size']
        self.sample_size = 2*self.trajectory_size

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.SAMPLE_IDX = None
        # IDX.shape: (batch, pomo)
        self.problems = None
        # shape: (batch, node, node)

        # Dynamic
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~problem)

    def load_problems(self, batch_size, aug_factor = None):

        self.problems = get_random_problems(batch_size, self.node_cnt, aug_factor)

        self.batch_size = self.problems.size(0)
        
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.sample_size)
        self.SAMPLE_IDX = torch.arange(self.sample_size)[None, :].expand(self.batch_size, self.sample_size)

    def load_problems_test(self, batch_size, aug_factor = None):

        self.problems = get_test_problems(batch_size, self.node_cnt, aug_factor)

        self.batch_size = self.problems.size(0)
        
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.sample_size)
        self.SAMPLE_IDX = torch.arange(self.sample_size)[None, :].expand(self.batch_size, self.sample_size)

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.sample_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~problem)

        # CREATE STEP STATE
        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, SAMPLE_IDX=self.SAMPLE_IDX)
        self.step_state.ninf_mask = torch.zeros((self.batch_size, self.sample_size, self.node_cnt))

        reward = None
        done = False
        return Reset_State(self.problems), reward, done

    def pre_step(self):
        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected,PPO_step = False):
        # selected.shape: (batch, pomo)

        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~problem)

        # UPDATE STEP STATE
        self.step_state.current_node = self.current_node
        # shape: (batch, pomo)
        self.step_state.ninf_mask[self.BATCH_IDX, self.SAMPLE_IDX, self.current_node] = float('-inf')
        # shape: (batch, pomo, node)

        # returning values
        done = (self.selected_count == self.node_cnt)
        if done:
            reward = -self._get_total_distance()  # note the minus sign!           
        else:
            reward = None


        return self.step_state, reward, done

    def _get_total_distance(self):
        
        node_from = self.selected_node_list
        # shape: (batch, pomo, node)
        node_to = node_from.roll(dims=2, shifts=-1)
        # shape: (batch, pomo, node)
        batch_index = self.BATCH_IDX[:, :, None].expand(self.batch_size, self.sample_size, self.node_cnt)
        # shape: (batch, pomo, node)

        selected_cost = self.problems[batch_index, node_from, node_to]
        # shape: (batch, pomo, node)
        total_distance = selected_cost.sum(2)
        # shape: (batch, pomo)

        return total_distance

