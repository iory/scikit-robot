import numpy as np
from . import utils

def plan_trajectory_rrt(self, 
        target_coords,
        link_list,
        end_effector
        ):
    joint_list = [link.joint for link in link_list]
    joint_mins = [max(j.min_angle, -3.14) for j in joint_list]
    joint_maxs = [min(j.max_angle, +3.14) for j in joint_list]
    cspace = ConfigurationSpace(joint_mins, joint_maxs)
    av_init = utils.get_robot_state(self, joint_list)
    def pred_goal_condition(av):
        position = utils.forward_kinematics(self, link_list, av, end_effector, False, False, False)
        dist = np.linalg.norm(target_coords.worldpos() - position)
        return dist < 0.1
    pred_valid_config = lambda q : True
    rrt = RapidlyExploringRandomTree(cspace, av_init, pred_goal_condition, pred_valid_config)
    for i in range(1000):
        print(rrt.extend())

class ConfigurationSpace(object):
    def __init__(self, b_min, b_max):
        self.b_min = np.array(b_min)
        self.b_max = np.array(b_max)
        self.n_dof = len(b_min)

    def sample(self):
        w = self.b_max - self.b_min
        return np.random.rand(self.n_dof) * w + self.b_min

class RapidlyExploringRandomTree(object): 
    def __init__(self, cspace, q_start, pred_goal_condition, pred_valid_config,
            N_maxiter=10000):
        self.cspace = cspace
        self.eps = 0.1
        self.n_resolution = 10
        self.N_maxiter = N_maxiter
        self.isValid = pred_valid_config
        self.isGoal = pred_goal_condition

        # reserving memory is the key 
        self.Q_sample = np.zeros((N_maxiter, cspace.n_dof))
        self.idxes_parents = np.zeros(N_maxiter, dtype='int64')

        # set initial sample
        self.n_sample = 1
        self.Q_sample[0] = q_start
        self.idxes_parents[0] = 0 # self reference

    @property
    def q_start(self):
        return self.X_sample[0]

    def extend(self):
        if self.n_sample ==  self.N_maxiter:
            raise Exception

        def unit_vec(vec):
            return vec/np.linalg.norm(vec)

        q_rand = self.cspace.sample()
        q_rand_copied = np.repeat(q_rand.reshape(1, -1), self.n_sample, axis=0)

        sqdists = np.sum((self.Q_sample[:self.n_sample] - q_rand_copied)**2, axis=1)
        idx_nearest = np.argmin(sqdists)
        q_nearest = self.Q_sample[idx_nearest]
        if np.linalg.norm(q_rand - q_nearest) > self.eps:
            q_new = q_nearest + unit_vec(q_rand - q_nearest) * self.eps
        else:
            q_new = q_rand

        # update tree
        q_new_reshaped = q_new.reshape(1, -1)
        if self.isValid(q_new_reshaped):
            self.Q_sample[self.n_sample] = q_new
            self.idxes_parents[self.n_sample] = idx_nearest
            self.n_sample += 1

        return self.isGoal(q_new)

    def show(self):
        fig, ax = plt.subplots()
        n = self.n_sample
        ax.scatter(self.Q_sample[:n, 0], self.Q_sample [:n, 1], c="black")
        for q, parent_idx in zip(self.Q_sample[:n], self.idxes_parents[:n]):
            q_parent = self.Q_sample[parent_idx]
            ax.plot([q[0], q_parent[0]], [q[1], q_parent[1]], color="red")

