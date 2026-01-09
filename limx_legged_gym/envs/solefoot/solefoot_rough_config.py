
from limx_legged_gym.envs.base.legged_robot_config import LeggedRobotCfgPPO
from limx_legged_gym.envs.base.base_config import BaseConfig

class SoleFootRoughCfg(BaseConfig):
    class env:
        num_envs = 2048
        num_observations = 36  # 3(ang_vel) + 3(gravity) + 8(dof_pos) + 8(dof_vel) + 8(actions) + 1(sin) + 1(cos) + 4(gait)
        num_height_samples = 117
        num_privileged_obs = 200 # 83 (explicit privileged) + 117 (heights)
        num_actions = 8
        env_spacing = 3.0
        send_timeouts = True
        episode_length_s = 20
        obs_history_length = 5
        dof_vel_use_pos_diff = True
        fail_to_terminal_time_s = 0.5

    class terrain:
        mesh_type = "trimesh"
        horizontal_scale = 0.1
        vertical_scale = 0.005
        border_size = 25
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        measure_heights = False
        critic_measure_heights = True
        measured_points_x = [-0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        measured_points_y = [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]
        selected = False
        terrain_kwargs = None
        max_init_terrain_level = 5
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10
        num_cols = 20
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        slope_treshold = 0.75

    class commands:
        curriculum = True
        max_curriculum = 1.
        num_commands = 4
        resampling_time = 10.
        heading_command = True
        zero_command_prob = 0.0
        smooth_max_lin_vel_x = 1.0
        smooth_max_lin_vel_y = 1.0
        non_smooth_max_lin_vel_x = 1.0
        non_smooth_max_lin_vel_y = 1.0
        class ranges:
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-1, 1]
            heading = [-3.14, 3.14]

    class init_state:
        pos = [0.0, 0.0, 0.75]
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]
        default_joint_angles = {
            "abad_L_Joint": 0.0,
            "hip_L_Joint": 0.58,
            "knee_L_Joint": 1.35,
            "abad_R_Joint": 0.0,
            "hip_R_Joint": -0.58,
            "knee_R_Joint": -1.35,
            "ankle_L_Joint": -0.8,
            "ankle_R_Joint": -0.8
        }

    class control:
        control_type = 'P'
        max_power = 200.0
        stiffness = {
            "abad_L_Joint": 45, "hip_L_Joint": 45, "knee_L_Joint": 45, "ankle_L_Joint": 45,
            "abad_R_Joint": 45, "hip_R_Joint": 45, "knee_R_Joint": 45, "ankle_R_Joint": 45,
        }
        damping = {
            "abad_L_Joint": 1.5, "hip_L_Joint": 1.5, "knee_L_Joint": 1.5, "ankle_L_Joint": 0.8,
            "abad_R_Joint": 1.5, "hip_R_Joint": 1.5, "knee_R_Joint": 1.5, "ankle_R_Joint": 0.8,
        }
        action_scale = 0.25
        decimation = 8
        user_torque_limit = 80.0

    class asset:
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/SF_TRON1A/urdf/robot.urdf"
        name = "solefoot"
        foot_name = "ankle"
        foot_radius = 0.02
        penalize_contacts_on = ["knee", "hip"]
        terminate_after_contacts_on = ["abad", "base"]
        disable_gravity = False
        collapse_fixed_joints = True
        fix_base_link = False
        default_dof_drive_mode = 3
        self_collisions = 0
        replace_cylinder_with_capsule = True
        flip_visual_attachments = False
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 1.]
        randomize_base_com = True
        rand_com_vec = [0.03, 0.02, 0.03]
        randomize_inertia = True
        randomize_inertia_range = [0.8, 1.2]
        randomize_restitution = True
        restitution_range = [0.0, 1.0]
        randomize_Kp = True
        randomize_Kp_range = [0.8, 1.2]
        randomize_Kd = True
        randomize_Kd_range = [0.8, 1.2]
        randomize_motor_torque = True
        randomize_motor_torque_range = [0.8, 1.2]
        randomize_default_dof_pos = True
        randomize_default_dof_pos_range = [-0.05, 0.05]
        randomize_action_delay = True
        delay_ms_range = [0, 20]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.

    class gait:
        num_gait_params = 4
        resampling_time = 5
        touch_down_vel = 0.0
        class ranges:
            frequencies = [1.5, 2.5]
            offsets = [0, 1]
            durations = [0.5, 0.5]
            swing_height = [0.0, 0.1]

    class rewards:
        ang_tracking_sigma = 0.25
        clip_single_reward = 5.0
        clip_reward = 100.0
        only_positive_rewards = True
        class scales:
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -5.0
            torques = -0.000025
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0.
            feet_air_time = 1.0
            collision = -1.
            feet_stumble = -0.0
            action_rate = -0.01
            stand_still = -0.
            action_smooth = -0.01
            keep_balance = 1.0
            tracking_lin_vel_pb = 1.0
            tracking_ang_vel_pb = 0.2
            tracking_contacts_shaped_force = -2.0
            tracking_contacts_shaped_vel = -2.0
            feet_distance = -0.2

        only_positive_rewards = True
        tracking_sigma = 0.25
        soft_dof_pos_limit = 1.
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.75
        max_contact_force = 100.
        kappa_gait_probs = 0.05
        gait_force_sigma = 25.0
        gait_vel_sigma = 0.25
        min_feet_distance = 0.2

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
            contact_forces = 0.01
            torque = 0.05
            dof_acc = 0.0025
        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = True
        noise_level = 1.0
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class viewer:
        ref_env = 0
        pos = [10, 0, 6]
        lookat = [11., 5, 3.]

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]
        up_axis = 1
        class physx:
            num_threads = 10
            solver_type = 1
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01
            rest_offset = 0.0
            bounce_threshold_velocity = 0.5
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2 ** 23
            default_buffer_size_multiplier = 5
            contact_collection = 2

class SoleFootRoughCfgPPO(LeggedRobotCfgPPO):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    save_interval = 500
    class policy:
        class actor:
            class encoder:
                class_name = 'ProprioceptiveObsEncoder'
                output_dim = 3
                hidden_dims = [256, 128]
                is_detach = True
            class_name = 'ParitialObsEncodedMlpActor'
            hidden_dims = [512, 256, 128]
            init_noise_std = 1.0
            activation = 'elu'
        class critic:
            class_name = 'MlpCritic'
            hidden_dims = [512, 256, 128]
            activation = 'elu'
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
    class runner(LeggedRobotCfgPPO.runner):
        max_iterations = 10000
        experiment_name = 'solefoot_rough'
