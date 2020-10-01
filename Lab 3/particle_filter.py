from grid import CozGrid
from particle import Particle
from utils import grid_distance, rotate_point, diff_heading_deg, add_odometry_noise
import setting
import math
import numpy as np


def motion_update(particles, odom):
    """ Particle filter motion update

        Arguments:
        particles -- input list of particle represents belief p(x_{t-1} | u_{t-1})
                before motion update
        odom -- odometry to move (dx, dy, dh) in *robot local frame*

        Returns: the list of particles represents belief \tilde{p}(x_{t} | u_{t})
                after motion update
    """
    motion_particles = particles
    if odom[0] == 0 and odom[1] == 0 and odom[2] == 0:
        return motion_particles
    for i in range(len(motion_particles)):
        particle = motion_particles[i]
        p_x, p_y, p_h = particle.xyh
        n_x, n_y, d_h = add_odometry_noise(odom, setting.ODOM_HEAD_SIGMA, setting.ODOM_TRANS_SIGMA)
        d_x, d_y = rotate_point(n_x, n_y, p_h)

        np = Particle(p_x + d_x, p_y + d_y, p_h + d_h)
        motion_particles[i] = np

    return motion_particles

# ------------------------------------------------------------------------
def measurement_update(particles, measured_marker_list, grid):
    """ Particle filter measurement update

        Arguments:
        particles -- input list of particle represents belief \tilde{p}(x_{t} | u_{t})
                before measurement update (but after motion update)

        measured_marker_list -- robot detected marker list, each marker has format:
                measured_marker_list[i] = (rx, ry, rh)
                rx -- marker's relative X coordinate in robot's frame
                ry -- marker's relative Y coordinate in robot's frame
                rh -- marker's relative heading in robot's frame, in degree

                * Note that the robot can only see markers which is in its camera field of view,
                which is defined by ROBOT_CAMERA_FOV_DEG in setting.py
				* Note that the robot can see mutliple markers at once, and may not see any one

        grid -- grid world map, which contains the marker information,
                see grid.py and CozGrid for definition
                Can be used to evaluate particles

        Returns: the list of particles represents belief p(x_{t} | u_{t})
                after measurement update
    """
    measured_particles = particles
    return measured_particles


