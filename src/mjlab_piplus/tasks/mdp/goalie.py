def goal_pos_b(env, goal_pos=(-0.2,0.0) ):

# Into the base frame.
robot = env.scene["robot"]
goal = + env_origins
    base_xy = robot.data.root_link_pos_w[:, :2]
    heading = robot.data.heading_w
    cos_h, sin_h = torch.cos(heading), torch.sin(heading)
    delta = pos_w - base_xy
    
    
    
    