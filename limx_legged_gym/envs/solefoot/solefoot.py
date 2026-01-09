from limx_legged_gym.envs.pointfoot.point_foot import PointFoot
from limx_legged_gym.utils.math import CubicSpline
import torch
import numpy as np

class SoleFoot(PointFoot):
    def _init_buffers(self):
        super()._init_buffers()
        self.des_foot_height = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.des_foot_velocity_z = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        
        # 3 (ang_vel) + 3 (gravity) + 8 (dof_pos) + 8 (dof_vel) + 8 (actions) + ...
        noise_vec[0:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:14] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[14:22] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[22:30] = 0.0 # previous actions
        # clock inputs and gaits (indices 30:36) have no noise
        
        return noise_vec

    def _step_contact_targets(self):
        super()._step_contact_targets()
        self._generate_des_ee_ref()

    def _generate_des_ee_ref(self):
        frequencies = self.gaits[:, 0]
        mask_0 = (self.gait_indices < 0.25) & (self.gait_indices >= 0.0)  # lift up
        mask_1 = (self.gait_indices < 0.5) & (self.gait_indices >= 0.25)  # touch down
        mask_2 = (self.gait_indices < 0.75) & (self.gait_indices >= 0.5)  # lift up
        mask_3 = (self.gait_indices <= 1.0) & (self.gait_indices >= 0.75)  # touch down
        swing_start_time = torch.zeros(self.num_envs, device=self.device)
        swing_start_time[mask_1] = 0.25 / frequencies[mask_1]
        swing_start_time[mask_2] = 0.5 / frequencies[mask_2]
        swing_start_time[mask_3] = 0.75 / frequencies[mask_3]
        swing_end_time = swing_start_time + 0.25 / frequencies
        swing_start_pos = torch.ones(self.num_envs, device=self.device)
        swing_start_pos[mask_0] = 0.0
        swing_start_pos[mask_2] = 0.0
        swing_end_pos = torch.ones(self.num_envs, device=self.device)
        swing_end_pos[mask_1] = 0.0
        swing_end_pos[mask_3] = 0.0
        swing_end_vel = torch.ones(self.num_envs, device=self.device)
        swing_end_vel[mask_0] = 0.0
        swing_end_vel[mask_2] = 0.0
        swing_end_vel[mask_1] = self.cfg.gait.touch_down_vel
        swing_end_vel[mask_3] = self.cfg.gait.touch_down_vel

        # generate desire foot z trajectory
        swing_height = self.gaits[:, 3]

        start = {'time': swing_start_time, 'position': swing_start_pos * swing_height,
                 'velocity': torch.zeros(self.num_envs, device=self.device)}
        end = {'time': swing_end_time, 'position': swing_end_pos * swing_height,
               'velocity': swing_end_vel}
        cubic_spline = CubicSpline(start, end)
        self.des_foot_height = cubic_spline.position(self.gait_indices / frequencies)
        self.des_foot_velocity_z = cubic_spline.velocity(self.gait_indices / frequencies)

    # Rewards from BipedSF
    def _reward_tracking_contacts_shaped_force(self):
        foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        desired_contact = self.desired_contact_states

        reward = 0
        if self.reward_scales["tracking_contacts_shaped_force"] > 0:
            for i in range(len(self.feet_indices)):
                swing_phase = 1 - desired_contact[:, i]
                reward += swing_phase * torch.exp(
                    -foot_forces[:, i] ** 2 / self.cfg.rewards.gait_force_sigma
                )
        else:
            for i in range(len(self.feet_indices)):
                swing_phase = 1 - desired_contact[:, i]
                reward += swing_phase * (
                    1
                    - torch.exp(
                        -foot_forces[:, i] ** 2 / self.cfg.rewards.gait_force_sigma
                    )
                )
        return reward / len(self.feet_indices)

    def _reward_tracking_contacts_shaped_vel(self):
        foot_velocities = torch.norm(self.foot_velocities, dim=-1)
        desired_contact = self.desired_contact_states
        reward = 0
        if self.reward_scales["tracking_contacts_shaped_vel"] > 0:
            for i in range(len(self.feet_indices)):
                stand_phase = desired_contact[:, i]
                reward += stand_phase * torch.exp(
                    -foot_velocities[:, i] ** 2 / self.cfg.rewards.gait_vel_sigma
                )
                swing_phase = 1 - desired_contact[:, i]
                reward += swing_phase * torch.exp(
                    -((self.foot_velocities[:, i, 2] - self.des_foot_velocity_z) ** 2)
                    / self.cfg.rewards.gait_vel_sigma
                )
        else:
            for i in range(len(self.feet_indices)):
                stand_phase = desired_contact[:, i]
                reward += stand_phase * (
                    1
                    - torch.exp(
                        -foot_velocities[:, i] ** 2 / self.cfg.rewards.gait_vel_sigma
                    )
                )
                swing_phase = 1 - desired_contact[:, i]
                reward += swing_phase * (1 - torch.exp(
                    -((self.foot_velocities[:, i, 2] - self.des_foot_velocity_z) ** 2)
                    / self.cfg.rewards.gait_vel_sigma)
                )
        return reward / len(self.feet_indices)
