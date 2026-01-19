
from dataclasses import dataclass
import torch

from PFSProblemDef import get_random_problems
from PFSProblemDefTest import get_random_problems as get_random_problems_test 


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


class PFSPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.job_size = env_params['job_size']
        self.machine_size = env_params['machine_size']
        self.sample_size = 2*env_params['trajectory_size']
        self.trajectory_size = env_params['trajectory_size']

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

    def load_problems(self, batch_size,aug_factor=1):
        self.batch_size = batch_size
        
        self.problems = get_random_problems(self.batch_size,self.job_size, self.machine_size)

        # problems.shape: (batch, problem, 2)
        if aug_factor > 1:
            self.batch_size = self.batch_size * 2
            self.problems = augment_PFSP(self.problems)

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.sample_size)
        self.SAMPLE_IDX = torch.arange(self.sample_size)[None, :].expand(self.batch_size, self.sample_size)
    
    def load_problems_test(self, batch_size):
        self.batch_size = batch_size
        
        self.problems = get_random_problems_test(self.batch_size,self.job_size, self.machine_size)

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
        self.step_state.ninf_mask = torch.zeros((self.batch_size, self.sample_size, self.job_size))

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
        done = (self.selected_count == self.job_size)
        if done:
            reward = -self._get_makespan()  # note the minus sign!
            route = self.selected_node_list  
        else:
            reward = None
            route = None

        return self.step_state, reward, done, route

    def _get_makespan(self):

        # selected_node_Forward = self.selected_node_list[:,:self.trajectory_size]
        # selected_node_Backward = self.selected_node_list[:,self.trajectory_size:].flip(dims = (2,))
        selected_node = self.selected_node_list

        # 각 머신에서 작업이 끝나는 시간을 저장하는 배열
        machine_end_times = torch.zeros(self.batch_size, self.sample_size, self.machine_size)

        # 각 작업이 끝나는 시간을 저장하는 배열
        job_end_times = torch.zeros(self.batch_size, self.sample_size, self.job_size)

        for j in range(self.job_size):
            current_node = selected_node[self.BATCH_IDX, self.SAMPLE_IDX, j]
            for m in range(self.machine_size):
                if m == 0:  # 첫 번째 머신에서는 이전 작업의 완료 시간에 관계없이 처리
                    start_time = machine_end_times[self.BATCH_IDX, self.SAMPLE_IDX, m]  # 이전 작업의 종료 시간
                else:  # 그 외의 머신에서는 이전 머신과의 종속 관계가 있음
                    start_time = torch.max(machine_end_times[self.BATCH_IDX, self.SAMPLE_IDX, m],
                                           job_end_times[self.BATCH_IDX, self.SAMPLE_IDX, current_node])

                job_end_times[self.BATCH_IDX, self.SAMPLE_IDX, current_node] = start_time + self.problems[
                    self.BATCH_IDX, current_node, m]
                machine_end_times[self.BATCH_IDX, self.SAMPLE_IDX, m] = job_end_times[
                    self.BATCH_IDX, self.SAMPLE_IDX, current_node]

        # 마지막 머신에서의 최종 작업이 끝나는 시간 = makespan
        makespan = machine_end_times[:, :, -1]

        return makespan

