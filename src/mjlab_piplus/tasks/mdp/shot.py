 # Into the base frame.
    base_xy = self.robot.data.root_link_pos_w[:, :2]
    heading = self.robot.data.heading_w
    cos_h, sin_h = torch.cos(heading), torch.sin(heading)
    delta = pos_w - base_xy
    self.ball_pos_b = torch.stack(
      [
        cos_h * delta[:, 0] + sin_h * delta[:, 1],
        -sin_h * delta[:, 0] + cos_h * delta[:, 1],
      ],
      dim=1,
    )
    self.ball_vel_b = torch.stack(
      [
        cos_h * vel_w[:, 0] + sin_h * vel_w[:, 1],
        -sin_h * vel_w[:, 0] + cos_h * vel_w[:, 1],
      ],
      dim=1,
    )
    self.dir_err = wrap_to_pi(self.kick_dir_w - heading)

    self._update_approach_geometry(cos_h, sin_h)
    self._update_cycle(active, vel_w)