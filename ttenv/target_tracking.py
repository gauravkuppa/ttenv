"""Target Tracking Environments for Reinforcement Learning. OpenAI gym format

[Vairables]

d: radial coordinate of a belief target in the learner frame
alpha : angular coordinate of a belief target in the learner frame
ddot : radial velocity of a belief target in the learner frame
alphadot : angular velocity of a belief target in the learner frame
Sigma : Covariance of a belief target
o_d : linear distance to the closet obstacle point
o_alpha : angular distance to the closet obstacle point

[Environment Descriptions]

TargetTrackingEnv0 : Static Target model + noise - No Velocity Estimate
    RL state: [d, alpha, logdet(Sigma), observed] * nb_targets , [o_d, o_alpha]
    Target: Static [x,y] + noise
    Belief Target: KF, Estimate only x and y

TargetTrackingEnv1 : Double Integrator Target model with KF belief tracker
    RL state: [d, alpha, ddot, alphadot, logdet(Sigma), observed] * nb_targets, [o_d, o_alpha]
    Target : Nonlinear Double Integrator model, [x,y,xdot,ydot]
    Belief Target : KF, Double Integrator model

TargetTrackingEnv2 : SE2 Target model with UKF belief tracker
    RL state: [d, alpha, logdet(Sigma), observed] * nb_targets, [o_d, o_alpha]
    Target : SE2 model [x,y,theta] + a control policy u=[v,w]
    Belief Target : UKF for SE2 model [x,y,theta]

TargetTrackingEnv3 : SE2 Target model with UKF belief tracker [x,y,theta,v,w]
    RL state: [d, alpha, ddot, alphadot, logdet(Sigma), observed] * nb_targets, [o_d, o_alpha]
    Target : SE2 model [x,y,theta] + a control policy u=[v,w]
    Belief Target : UKF for SE2Vel model [x,y,theta,v,w]
"""
from gym import spaces, logger

import numpy as np
from numpy import linalg as LA

from ttenv.base import TargetTrackingBase
from ttenv.maps import map_utils
from ttenv.agent_models import *
from ttenv.policies import *
from ttenv.belief_tracker import KFbelief, UKFbelief
from ttenv.metadata import METADATA
import ttenv.util as util
from ttenv.base import TargetTrackingBase

class TargetTrackingEnv0(TargetTrackingBase):
    def __init__(self, num_targets=1, map_name='empty', is_training=True,
                                                    known_noise=True, **kwargs):
        TargetTrackingBase.__init__(self, num_targets=num_targets, map_name=map_name,
            is_training=is_training, known_noise=known_noise, **kwargs)
        self.id = 'TargetTracking-v0'
        self.target_dim = 2

        # Set limits.
        self.set_limits()

        # Build an agent, targets, and beliefs.
        self.build_models(const_q=METADATA['const_q'], known_noise=known_noise)

    def reset(self, **kwargs):
        if 'const_q' in kwargs:
            self.build_models(const_q=kwargs['const_q'])

        # Reset the agent, targets, and beliefs with sampled initial positions.
        init_pose = super().reset(**kwargs)
        self.agent.reset(init_pose['agent'])
        for i in range(self.num_targets):
            self.belief_targets[i].reset(
                    init_state=init_pose['belief_targets'][i][:self.target_dim],
                    init_cov=self.target_init_cov)
            self.targets[i].reset(np.array(init_pose['targets'][i][:self.target_dim]))

        # The targets are observed by the agent (z_0) and the beliefs are updated (b_0).
        observed = self.observe_and_update_belief()

        # Predict the target for the next step, b_1|0.
        self.belief_targets[i].predict()

        # Compute the RL state.
        self.state_func([0.0, 0.0], observed)
        return self.state

    def state_func(self, action_vw, observed):
        # Find the closest obstacle coordinate.
        obstacles_pt = self.MAP.get_closest_obstacle(self.agent.state)
        if obstacles_pt is None:
            obstacles_pt = (self.sensor_r, np.pi)

        self.state = []
        for i in range(self.num_targets):
            r_b, alpha_b = util.relative_distance_polar(
                                            self.belief_targets[i].state[:2],
                                                xy_base=self.agent.state[:2],
                                                theta_base=self.agent.state[2])
            self.state.extend([r_b, alpha_b,
                                    np.log(LA.det(self.belief_targets[i].cov)),
                                    float(observed[i])])
        self.state.extend([obstacles_pt[0], obstacles_pt[1]])
        self.state = np.array(self.state)

        # Update the visit map for the evaluation purpose.
        if self.MAP.visit_map is not None:
            self.MAP.update_visit_freq_map(self.agent.state, 1.0, observed=bool(np.mean(observed)))

    def set_limits(self):
        self.num_target_dep_vars = 4
        self.num_target_indep_vars = 2

        self.limit = {} # 0: low, 1:high
        self.limit['agent'] = [np.concatenate((self.MAP.mapmin,[-np.pi])), np.concatenate((self.MAP.mapmax, [np.pi]))]
        self.limit['target'] = [self.MAP.mapmin, self.MAP.mapmax]
        self.limit['state'] = [np.concatenate(([0.0, -np.pi, -50.0, 0.0]*self.num_targets, [0.0, -np.pi ])),
                               np.concatenate(([600.0, np.pi, 50.0, 2.0]*self.num_targets, [self.sensor_r, np.pi]))]
        self.observation_space = spaces.Box(self.limit['state'][0], self.limit['state'][1], dtype=np.float32)
        assert(len(self.limit['state'][0]) == (self.num_target_dep_vars * self.num_targets + self.num_target_indep_vars))

    def build_models(self, const_q=None, known_noise=True, **kwargs):
        if const_q is None:
            self.const_q = np.random.choice([0.001, 0.1, 1.0])
        else:
            self.const_q = const_q

        # Build a robot
        self.agent = AgentSE2(dim=3, sampling_period=self.sampling_period, limit=self.limit['agent'],
                            collision_func=lambda x: self.MAP.is_collision(x))

        self.target_noise_cov = self.const_q * self.sampling_period**3 / 3 * np.eye(self.target_dim)
        if known_noise:
            self.target_true_noise_sd = self.target_noise_cov
        else:
            self.target_true_noise_sd = METADATA['const_q_true'] * np.eye(2)
        self.targetA = np.eye(self.target_dim)

        # Build a target
        self.targets = [AgentDoubleInt2D(dim=self.target_dim, sampling_period=self.sampling_period,
                            limit=self.limit['target'],
                            collision_func=lambda x: self.MAP.is_collision(x),
                            A=self.targetA, W=self.target_true_noise_sd) for _ in range(self.num_targets)]
        self.belief_targets = [KFbelief(dim=self.target_dim, limit=self.limit['target'], A=self.targetA,
                            W=self.target_noise_cov, obs_noise_func=self.observation_noise,
                            collision_func=lambda x: self.MAP.is_collision(x))
                                for _ in range(self.num_targets)]

class TargetTrackingEnv1(TargetTrackingBase):
    def __init__(self, num_targets=1, map_name='empty', is_training=True, known_noise=True, **kwargs):
        TargetTrackingBase.__init__(self, num_targets=num_targets, map_name=map_name,
            is_training=is_training, known_noise=known_noise, **kwargs)
        self.id = 'TargetTracking-v1'
        self.target_dim = 4
        self.target_init_vel = np.array(METADATA['target_init_vel'])

        # Set limits.
        self.set_limits(target_speed_limit=METADATA['target_speed_limit'])

        # Build an agent, targets, and beliefs.
        self.build_models(const_q=METADATA['const_q'], known_noise=known_noise)

    def reset(self, **kwargs):
        # Always set the limits first.
        if 'target_speed_limit' in kwargs:
            self.set_limits(target_speed_limit=kwargs['target_speed_limit'])

        if 'const_q' in kwargs:
            self.build_models(const_q=kwargs['const_q'])

        # Reset the agent, targets, and beliefs with sampled initial positions.
        init_pose = super().reset(**kwargs)
        self.agent.reset(init_pose['agent'])
        for i in range(self.num_targets):
            self.belief_targets[i].reset(
                        init_state=np.concatenate((init_pose['belief_targets'][i][:2], np.zeros(2))),
                        init_cov=self.target_init_cov)
            self.targets[i].reset(np.concatenate((init_pose['targets'][i][:2], self.target_init_vel)))

        # The targets are observed by the agent (z_0) and the beliefs are updated (b_0).
        observed = self.observe_and_update_belief()

        # Predict the target for the next step, b_1|0.
        for i in range(self.num_targets):
            self.belief_targets[i].predict()

        # Compute the RL state.
        self.state_func([0.0, 0.0], observed)
        return self.state

    def state_func(self, action_vw, observed):
        # Find the closest obstacle coordinate.
        obstacles_pt = self.MAP.get_closest_obstacle(self.agent.state)
        if obstacles_pt is None:
            obstacles_pt = (self.sensor_r, np.pi)

        self.state = []
        for i in range(self.num_targets):
            r_b, alpha_b = util.relative_distance_polar(self.belief_targets[i].state[:2],
                                                xy_base=self.agent.state[:2],
                                                theta_base=self.agent.state[2])
            r_dot_b, alpha_dot_b = util.relative_velocity_polar(
                                    self.belief_targets[i].state[:2],
                                    self.belief_targets[i].state[2:],
                                    self.agent.state[:2], self.agent.state[2],
                                    action_vw[0], action_vw[1])
            self.state.extend([r_b, alpha_b, r_dot_b, alpha_dot_b,
                                    np.log(LA.det(self.belief_targets[i].cov)),
                                    float(observed[i])])
        self.state.extend([obstacles_pt[0], obstacles_pt[1]])
        self.state = np.array(self.state)

        # Update the visit map when there is any target not observed for the evaluation purpose.
        if self.MAP.visit_map is not None:
            self.MAP.update_visit_freq_map(self.agent.state, 1.0, observed=bool(np.mean(observed)))

    def set_limits(self, target_speed_limit=None):
        self.num_target_dep_vars = 6
        self.num_target_indep_vars = 2

        if target_speed_limit is None:
            self.target_speed_limit = np.random.choice([1.0, 3.0])
        else:
            self.target_speed_limit = target_speed_limit
        rel_speed_limit = self.target_speed_limit + METADATA['action_v'][0] # Maximum relative speed

        self.limit = {} # 0: low, 1:highs
        self.limit['agent'] = [np.concatenate((self.MAP.mapmin,[-np.pi])), np.concatenate((self.MAP.mapmax, [np.pi]))]
        self.limit['target'] = [np.concatenate((self.MAP.mapmin,[-self.target_speed_limit, -self.target_speed_limit])),
                                np.concatenate((self.MAP.mapmax, [self.target_speed_limit, self.target_speed_limit]))]
        self.limit['state'] = [np.concatenate(([0.0, -np.pi, -rel_speed_limit, -10*np.pi, -50.0, 0.0]*self.num_targets, [0.0, -np.pi])),
                               np.concatenate(([600.0, np.pi, rel_speed_limit, 10*np.pi,  50.0, 2.0]*self.num_targets, [self.sensor_r, np.pi]))]
        self.observation_space = spaces.Box(self.limit['state'][0], self.limit['state'][1], dtype=np.float32)
        assert(len(self.limit['state'][0]) == (self.num_target_dep_vars * self.num_targets + self.num_target_indep_vars))

    def build_models(self, const_q=None, known_noise=True, **kwargs):
        if const_q is None:
            self.const_q = np.random.choice([0.001, 0.1, 1.0])
        else:
            self.const_q = const_q

        # Build a robot
        self.agent = AgentSE2(dim=3, sampling_period=self.sampling_period, limit=self.limit['agent'],
                            collision_func=lambda x: self.MAP.is_collision(x))
        # Build targets
        self.targetA = np.concatenate((np.concatenate((np.eye(2),
                                        self.sampling_period*np.eye(2)), axis=1),
                                        [[0,0,1,0],[0,0,0,1]]))
        self.target_noise_cov = self.const_q * np.concatenate((
                            np.concatenate((self.sampling_period**3/3*np.eye(2),
                                self.sampling_period**2/2*np.eye(2)), axis=1),
                        np.concatenate((self.sampling_period**2/2*np.eye(2),
                                    self.sampling_period*np.eye(2)),axis=1) ))
        if known_noise:
            self.target_true_noise_sd = self.target_noise_cov
        else:
            self.target_true_noise_sd = self.const_q_true * np.concatenate((
                        np.concatenate((self.sampling_period**2/2*np.eye(2),
                                    self.sampling_period/2*np.eye(2)), axis=1),
                        np.concatenate((self.sampling_period/2*np.eye(2),
                                    self.sampling_period*np.eye(2)),axis=1) ))

        self.targets = [AgentDoubleInt2D_Nonlinear(self.target_dim,
                            self.sampling_period, self.limit['target'],
                            lambda x: self.MAP.is_collision(x),
                            W=self.target_true_noise_sd, A=self.targetA,
                            obs_check_func=lambda x: self.MAP.get_closest_obstacle(
                                x, fov=2*np.pi, r_max=10e2))
                            for _ in range(self.num_targets)]
        self.belief_targets = [KFbelief(dim=self.target_dim,
                            limit=self.limit['target'], A=self.targetA,
                            W=self.target_noise_cov,
                            obs_noise_func=self.observation_noise,
                            collision_func=lambda x: self.MAP.is_collision(x))
                            for _ in range(self.num_targets)]

class TargetTrackingEnv2(TargetTrackingEnv0):
    def __init__(self, num_targets=1, map_name='empty', is_training=True, known_noise=True, **kwargs):
        TargetTrackingEnv0.__init__(self, num_targets=num_targets,
            map_name=map_name, is_training=is_training, known_noise=known_noise, **kwargs)
        self.id = 'TargetTracking-v2'
        self.target_dim = 3

        # Set limits.
        self.set_limits()

        # Build an agent, targets, and beliefs.
        self.build_models(const_q=METADATA['const_q'], known_noise=known_noise)

    def set_limits(self, target_speed_limit=None):
        self.num_target_dep_vars = 4
        self.num_target_indep_vars = 2

        if target_speed_limit is None:
            self.target_speed_limit = np.random.choice([1.0, 3.0])
        else:
            self.target_speed_limit = target_speed_limit

        # LIMIT
        self.limit = {} # 0: low, 1:highs
        self.limit['agent'] = [np.concatenate((self.MAP.mapmin,[-np.pi])), np.concatenate((self.MAP.mapmax, [np.pi]))]
        self.limit['target'] = [np.concatenate((self.MAP.mapmin, [-np.pi])), np.concatenate((self.MAP.mapmax, [np.pi]))]
        self.limit['state'] = [np.concatenate(([0.0, -np.pi, -50.0, 0.0]*self.num_targets, [0.0, -np.pi ])),
                               np.concatenate(([600.0, np.pi, 50.0, 2.0]*self.num_targets, [self.sensor_r, np.pi]))]
        self.observation_space = spaces.Box(self.limit['state'][0], self.limit['state'][1], dtype=np.float32)
        assert(len(self.limit['state'][0]) == (self.num_target_dep_vars * self.num_targets + self.num_target_indep_vars))

    def build_models(self, const_q=None, known_noise=True, **kwargs):
        if const_q is None:
            self.const_q = np.random.choice([0.001, 0.1, 1.0])
        else:
            self.const_q = const_q

        # Build a robot
        self.agent = AgentSE2(3, self.sampling_period, self.limit['agent'],
                            lambda x: self.MAP.is_collision(x))
        # Build a target
        self.targets = [AgentSE2(self.target_dim, self.sampling_period,
                        self.limit['target'],
                        lambda x: self.MAP.is_collision(x),
                        policy=SinePolicy(0.1, 0.5, 5.0, self.sampling_period))
                        for _ in range(self.num_targets)]

        self.target_noise_cov = self.const_q * self.sampling_period * np.eye(self.target_dim)
        if known_noise:
            self.target_true_noise_sd = self.target_noise_cov
        else:
            self.target_true_noise_sd = self.const_q_true * \
                                self.sampling_period * np.eye(self.target_dim)
        # SinePolicy(0.5, 0.5, 2.0, self.sampling_period)
        # CirclePolicy(self.sampling_period, self.MAP.origin, 3.0)
        # RandomPolicy()

        self.belief_targets = [UKFbelief(dim=self.target_dim,
                            limit=self.limit['target'], fx=SE2Dynamics,
                            W=self.target_noise_cov,
                            obs_noise_func=self.observation_noise,
                            collision_func=lambda x: self.MAP.is_collision(x))
                            for _ in range(self.num_targets)]

class TargetTrackingEnv3(TargetTrackingBase):
    def __init__(self, num_targets=1, map_name='empty', is_training=True, known_noise=True, **kwargs):
        TargetTrackingEnv0.__init__(self, num_targets=num_targets,
            map_name=map_name, is_training=is_training, known_noise=known_noise, **kwargs)
        self.id = 'TargetTracking-v3'
        self.target_dim = 5
        self.target_init_vel = np.array(METADATA['target_init_vel'])

        # Set limits.
        self.set_limits()

        # Build an agent, targets, and beliefs.
        self.build_models(const_q=METADATA['const_q'], known_noise=known_noise)

    def set_limits(self, target_speed_limit=None):
        if self.target_dim != 5:
            return
        self.num_target_dep_vars = 6
        self.num_target_indep_vars = 2

        if target_speed_limit is None:
            self.target_speed_limit = np.random.choice([1.0, 3.0])
        else:
            self.target_speed_limit = target_speed_limit
        rel_speed_limit = self.target_speed_limit + METADATA['action_v'][0] # Maximum relative speed

        # LIMIT
        self.limit = {} # 0: low, 1:highs
        rel_speed_limit = self.target_speed_limit + METADATA['action_v'][0] # Maximum relative speed
        self.limit['agent'] = [np.concatenate((self.MAP.mapmin,[-np.pi])), np.concatenate((self.MAP.mapmax, [np.pi]))]
        self.limit['target'] = [np.concatenate((self.MAP.mapmin, [-np.pi, -self.target_speed_limit, -np.pi])),
                                            np.concatenate((self.MAP.mapmax, [np.pi, self.target_speed_limit, np.pi]))]
        self.limit['state'] = [np.concatenate(([0.0, -np.pi, -rel_speed_limit, -10*np.pi, -50.0, 0.0]*self.num_targets, [0.0, -np.pi ])),
                               np.concatenate(([600.0, np.pi, rel_speed_limit, 10*np.pi, 50.0, 2.0]*self.num_targets, [self.sensor_r, np.pi]))]
        self.observation_space = spaces.Box(self.limit['state'][0], self.limit['state'][1], dtype=np.float32)

    def build_models(self, const_q=None, known_noise=True, **kwargs):
        if self.target_dim != 5:
            return

        if const_q is None:
            self.const_q = np.random.choice([0.001, 0.1, 1.0])
        else:
            self.const_q = const_q

        self.target_noise_cov = np.zeros((self.target_dim, self.target_dim))
        for i in range(3):
            self.target_noise_cov[i,i] = self.const_q * self.sampling_period**3/3
        self.target_noise_cov[3:, 3:] = self.const_q * \
                    np.array([[self.sampling_period, self.sampling_period**2/2],
                             [self.sampling_period**2/2, self.sampling_period]])
        if known_noise:
            self.target_true_noise_sd = self.target_noise_cov
        else:
            self.target_true_noise_sd = self.const_q_true * \
                                  self.sampling_period * np.eye(self.target_dim)
        # Build a robot
        self.agent = AgentSE2(3, self.sampling_period, self.limit['agent'],
                            lambda x: self.MAP.is_collision(x))
        # Build a target
        self.targets = [AgentSE2(self.target_dim, self.sampling_period, self.limit['target'],
                        lambda x: self.MAP.is_collision(x),
                        policy=ConstantPolicy(self.target_noise_cov[3:, 3:]))
                        for _ in range(self.num_targets)]
        # SinePolicy(0.5, 0.5, 2.0, self.sampling_period)
        # CirclePolicy(self.sampling_period, self.MAP.origin, 3.0)
        # RandomPolicy()

        self.belief_targets = [UKFbelief(dim=self.target_dim,
                            limit=self.limit['target'], fx=SE2DynamicsVel,
                            W=self.target_noise_cov,
                            obs_noise_func=self.observation_noise,
                            collision_func=lambda x: self.MAP.is_collision(x))
                            for _ in range(self.num_targets)]

    def reset(self, **kwargs):
        # Always set the limits first.
        if 'target_speed_limit' in kwargs:
            self.set_limits(target_speed_limit=kwargs['target_speed_limit'])

        if 'const_q' in kwargs:
            self.build_models(const_q=kwargs['const_q'])

        # Reset the agent, targets, and beliefs with sampled initial positions.
        init_pose = super().reset(**kwargs)
        self.agent.reset(init_pose['agent'])
        for i in range(self.num_targets):
            self.belief_targets[i].reset(
                        init_state=np.concatenate((init_pose['belief_targets'][i], np.zeros(2))),
                        init_cov=self.target_init_cov)
            t_init = np.concatenate((init_pose['targets'][i], [self.target_init_vel[0], 0.0]))
            self.targets[i].reset(t_init)
            self.targets[i].policy.reset(t_init)

        # The targets are observed by the agent (z_0) and the beliefs are updated (b_0).
        observed = self.observe_and_update_belief()

        # Predict the target for the next step, b_1|0.
        for i in range(self.num_targets):
            self.belief_targets[i].predict()

        # Compute the RL state.
        self.state_func([0.0, 0.0], observed)
        return self.state

    def state_func(self, action_vw, observed):
        # Find the closest obstacle coordinate.
        obstacles_pt = self.MAP.get_closest_obstacle(self.agent.state)
        if obstacles_pt is None:
            obstacles_pt = (self.sensor_r, np.pi)

        self.state = []
        for i in range(self.num_targets):
            r_b, alpha_b = util.relative_distance_polar(self.belief_targets[i].state[:2],
                                                xy_base=self.agent.state[:2],
                                                theta_base=self.agent.state[2])
            r_dot_b, alpha_dot_b = util.relative_velocity_polar_se2(
                                    self.belief_targets[i].state[:3],
                                    self.belief_targets[i].state[3:],
                                    self.agent.state, action_vw)
            self.state.extend([r_b, alpha_b, r_dot_b, alpha_dot_b,
                                    np.log(LA.det(self.belief_targets[i].cov)), float(observed[i])])
        self.state.extend([obstacles_pt[0], obstacles_pt[1]])
        self.state = np.array(self.state)
        # Update the visit map when there is any target not observed for the evaluation purpose.
        if self.MAP.visit_map is not None:
            self.MAP.update_visit_freq_map(self.agent.state, 1.0, observed=bool(np.mean(observed)))

class TargetTrackingEnvMultiAgent(TargetTrackingBase):
    def __init__(self, num_targets=2, num_agents=2, map_name='empty', is_training=True, known_noise=True, **kwargs):
        TargetTrackingEnv0.__init__(self, num_targets=num_targets,
            map_name=map_name, is_training=is_training, known_noise=known_noise, **kwargs)
        self.id = 'TargetTracking-MultiAgent'
        self.target_dim = 5
        self.target_init_vel = np.array(METADATA['target_init_vel'])
        self.num_agents = num_agents
        # Set limits.   
        self.set_limits()

        # Build an agent, targets, and beliefs.
        self.build_models(const_q=METADATA['const_q'], known_noise=known_noise)
    
    def set_limits(self, target_speed_limit=None):
        if self.target_dim != 5: # TODO: why?
            return
        
        self.num_target_dep_vars = 6
        self.num_target_indep_vars = 2

        if target_speed_limit is None:
            self.target_speed_limit = np.random.choice([1.0, 3.0])
        else:
            self.target_speed_limit = target_speed_limit
        self.limit = {} # 0: low, 1:highs
        rel_speed_limit = self.target_speed_limit + METADATA['action_v'][0]
        self.limit['agent'] = [np.concatenate((self.MAP.mapmin,[-np.pi])), np.concatenate((self.MAP.mapmax, [np.pi]))]
        self.limit['target'] = [np.concatenate((self.MAP.mapmin, [-np.pi, -self.target_speed_limit, -np.pi])),
                                            np.concatenate((self.MAP.mapmax, [np.pi, self.target_speed_limit, np.pi]))]
        self.limit['state'] = [np.concatenate(([0.0, -np.pi, -rel_speed_limit, -10*np.pi, -50.0, 0.0]*self.num_targets, [0.0, -np.pi ])),
                               np.concatenate(([600.0, np.pi, rel_speed_limit, 10*np.pi, 50.0, 2.0]*self.num_targets, [self.sensor_r, np.pi]))]
        self.observation_space = spaces.Box(self.limit['state'][0], self.limit['state'][1], dtype=np.float32)

    def build_models(self, const_q=None, known_noise=True, **kwargs):
        if self.target_dim != 5:
            return
        if const_q is None:
            self.const_q = np.random.choice([0.001, 0.1, 1.0])
        else:
            self.const_q = const_q
        
        self.target_noise_cov = np.zeros((self.target_dim, self.target_dim))
        for i in range(3):
            self.target_noise_cov[i, i] = self.const_q * self.sampling_period**3/3
        self.target_noise_cov[3:, 3:] = self.const_q * \
                    np.array([[self.sampling_period, self.sampling_period**2/2],
                        [self.sampling_period**2/2, self.sampling_period]])
        if known_noise:
            self.target_true_noise_sd = self.target_noise_cov
        else:
            self.target_true_noise_sd = self.const_q_true * \
                        self.sampling_period * np.eye(self.target_dim)
        
        self.agent = [AgentSE2(3, self.sampling_period, self.limit['agent'],
                            lambda x: self.MAP.is_collision(x)) 
                            for _ in range(self.num_agents)]
        self.targets = [AgentSE2(self.target_dim, self.sampling_period, self.limit['target'],
                        lambda x: self.MAP.is_collision(x),
                        policy=ConstantPolicy(self.target_noise_cov[3:, 3:]))
                        for _ in range(self.num_targets)]
       
        ## two dimensional array: self.belief_targets[0..num_agents][0..num_targets]
        self.belief_targets = [[UKFbelief(dim=self.target_dim,
                            limit=self.limit['target'], fx=SE2DynamicsVel,
                            W=self.target_noise_cov,
                            obs_noise_func=self.observation_noise,
                            collision_func=lambda x: self.MAP.is_collision(x))
                            for _ in range(self.num_targets)]
                            for _ in range(self.num_agents)]
        
    def step(self, action):
        # The agent performs an action (t -> t+1)
        action_vw = self.action_map[action]
        for i in range(self.num_agents):
            is_col = self.agent[i].update(action_vw, [t.state[:2] for t in self.targets])
            self.num_collisions += int(is_col)

        # The targets move (t -> t+1)
        for i in range(self.num_agents):
            for j in range(self.num_targets):
                if self.has_discovered[i][j]:
                    self.targets[i][j].update(self.agent[i].state[:2])

        # The targets are observed by the agent (z_t+1) and the beliefs are updated.
        observed = self.observe_and_update_belief()

        # Compute a reward from b_t+1|t+1 or b_t+1|t.
        reward, done, mean_nlogdetcov, std_nlogdetcov = self.get_reward(self.is_training,
                                                                is_col=is_col)
        # Predict the target for the next step, b_t+2|t+1
        for i in range(self.num_agents):
            for j in range(self.num_targets):
                self.belief_targets[i][j].predict()

        # Compute the RL state.
        self.state_func(action_vw, observed)

        return self.state, reward, done, {'mean_nlogdetcov': mean_nlogdetcov, 'std_nlogdetcov': std_nlogdetcov}

    def reset(self, **kwargs):
        # Always set the limits first.
        if 'target_speed_limit' in kwargs:
            self.set_limits(target_speed_limit=kwargs['target_speed_limit'])

        if 'const_q' in kwargs:
            self.build_models(const_q=kwargs['const_q'])

        # Reset the agent, targets, and beliefs with sampled initial positions.
        import pdb; pdb.set_trace()
        #init_pose = super().reset(**kwargs)
        init_pose = self.action_mapget_init_pose_random()
        for i in range(self.num_agents):
            self.agent[i].reset(list(np.array(init_pose['agent']) + np.random.random(3)))
        #from IPython import embed; embed()
        self.has_discovered = [[0] * self.num_targets for _ in range(self.num_agents)] # Set to 0 values for your evaluation purpose.
        
        for i in range(self.num_agents):
            for j in range(self.num_targets):
                self.belief_targets[i][j].reset(
                            init_state=np.concatenate((init_pose['belief_targets'][i], np.zeros(2))),
                            init_cov=self.target_init_cov)
                t_init = np.concatenate((init_pose['targets'][i], [self.target_init_vel[0], 0.0]))
                self.targets[i].reset(t_init)
                self.targets[i].policy.reset(t_init)

        # The targets are observed by the agent (z_0) and the beliefs are updated (b_0).
        observed = self.observe_and_update_belief()

        # Predict the target for the next step, b_1|0.
        for i in range(self.num_agents):
            for j in range(self.num_targets):
                self.belief_targets[i][j].predict()

        # Compute the RL state.
        self.state_func([0.0, 0.0], observed) # TODO: what should this be? 
        return self.state

    def get_init_pose_random(self,
                            lin_dist_range_a2b=METADATA['lin_dist_range_a2b'],
                            ang_dist_range_a2b=METADATA['ang_dist_range_a2b'],
                            lin_dist_range_b2t=METADATA['lin_dist_range_b2t'],
                            ang_dist_range_b2t=METADATA['ang_dist_range_b2t'],
                            blocked=None,
                            **kwargs):
        is_agent_valid = False
        while(not is_agent_valid):
            init_pose = {}
            init_pose['agent'] = []
            is_agent_valid = False
            for i in range(self.num_agents):
                while(not is_agent_valid):
                    a_init = np.random.random((2,)) * (self.MAP.mapmax-self.MAP.mapmin) + self.MAP.mapmin
                    is_agent_valid = not(self.MAP.is_collision(a_init))
                init_pose['agent'].append([a_init[0], a_init[1], np.random.random() * 2 * np.pi - np.pi])
            
            assert len(init_pose) == self.num_agents
            init_pose['targets'], init_pose['belief_targets'] = [], []
            for i in range(self.num_targets):
                count, is_belief_valid = 0, False
                while(not is_belief_valid):
                    is_belief_valid, init_pose_belief = self.gen_rand_pose(
                        init_pose['agent'][:2], init_pose['agent'][2],
                        lin_dist_range_a2b[0], lin_dist_range_a2b[1],
                        ang_dist_range_a2b[0], ang_dist_range_a2b[1])
                    if is_belief_valid and (blocked is not None):
                        is_blocked = self.MAP.is_blocked(init_pose['agent'][:2], init_pose_belief[:2])
                        is_belief_valid = (blocked == is_blocked)
                    count += 1
                    if count > 100:
                        is_agent_valid = False
                        break
                init_pose['belief_targets'].append(init_pose_belief)

                count, is_target_valid, init_pose_target = 0, False, np.zeros((2,))
                while((not is_target_valid) and is_belief_valid):
                    is_target_valid, init_pose_target = self.gen_rand_pose(
                        init_pose['belief_targets'][i][:2],
                        init_pose['belief_targets'][i][2],
                        lin_dist_range_b2t[0], lin_dist_range_b2t[1],
                        ang_dist_range_b2t[0], ang_dist_range_b2t[1])
                    if is_target_valid and (blocked is not None):
                        is_blocked = self.MAP.is_blocked(init_pose['agent'][:2], init_pose_target[:2])
                        is_target_valid = (blocked == is_blocked)
                    count += 1
                    if count > 100:
                        is_agent_valid = False
                        break
                init_pose['targets'].append(init_pose_target)
        return init_pose
        
    def state_func(self, action_vw, observed):
        # Find the closest obstacle coordinate.
        #obstacles_pt = self.MAP.get_closest_obstacle(self.agent.state)
        #if obstacles_pt is None:
        #    obstacles_pt = (self.sensor_r, np.pi)

        self.state = []
        # TODO: find distance between multiple agents and multiple targets
        for i in range(self.num_agents):
            for j in range(self.num_targets):
                r_b, alpha_b = util.relative_distance_polar(self.belief_targets[i][j].state[:2],
                                                    xy_base=self.agent[j].state[:2],
                                                    theta_base=self.agent[j].state[2])
                r_dot_b, alpha_dot_b = util.relative_velocity_polar_se2(
                                        self.belief_targets[i][j].state[:3],
                                        self.belief_targets[i][j].state[3:],
                                        self.agent[j].state, action_vw)
                self.state.extend([r_b, alpha_b, r_dot_b, alpha_dot_b,
                                        np.log(LA.det(self.belief_targets[i][j].cov)), float(observed[i])])
        #self.state.extend([obstacles_pt[0], obstacles_pt[1]])
        self.state = np.array(self.state)
        # Update the visit map when there is any target not observed for the evaluation purpose.
        if self.MAP.visit_map is not None:
            for i in range(self.num_agents):
                self.MAP.update_visit_freq_map(self.agent[i].state, 1.0, observed=bool(np.mean(observed)))

    def observation(self, target, agent):
        r, alpha = util.relative_distance_polar(target.state[:2],
                                            xy_base=agent.state[:2],
                                            theta_base=agent.state[2])
        observed = (r <= self.sensor_r) \
                    & (abs(alpha) <= self.fov/2/180*np.pi) \
                    & (not(self.MAP.is_blocked(agent.state, target.state)))
        z = None
        if observed:
            z = np.array([r, alpha])
            z += np.random.multivariate_normal(np.zeros(2,), self.observation_noise(z))
        return observed, z

    def observation_noise(self, z):
        obs_noise_cov = np.array([[self.sensor_r_sd * self.sensor_r_sd, 0.0],
                                [0.0, self.sensor_b_sd * self.sensor_b_sd]])
        return obs_noise_cov

    def observe_and_update_belief(self):
        observed = []
        for i in range(self.num_agents):
            for j in range(self.num_targets):
                observation = self.observation(self.targets[j], self.agent[i])
                observed.append(observation[0])
                if observation[0]: # if observed, update the target belief.
                    self.belief_targets[i][j].update(observation[1], self.agent[i].state)
                    if not(self.has_discovered[i][j]):
                        self.has_discovered[i][j] = 1
        return observed