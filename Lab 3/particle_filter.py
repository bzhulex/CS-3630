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

    d_x, d_y, d_h = odom

    if d_x == 0 and d_y == 0 and d_h == 0:
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


    def update_weight(landmarks, particle_readings):

        const_d = 2 * setting.MARKER_TRANS_SIGMA ** 2
        const_a = 2 * setting.MARKER_ROT_SIGMA ** 2

        if len(particle_readings) == 0:
            return 0 # if particle sees no landmarks, that particle gets a 0

        elif len(particle_readings) < len(measured_marker_list):
            return 0 # robot can't see more than particles which can see more

        else:
            # adjust probability w/ distance of robot_reading vs particle reading
            # keep the best one

            prob = 1.0
            for landmark in measured_marker_list:
                best_particle = None
                best_prob = float('-inf')
                for particle_reading in particle_readings:

                    d = math.sqrt(landmark[0]**2 + landmark[1]**2) - math.sqrt(particle_reading[0]**2 + particle_reading[1]**2)
                    a = diff_heading_deg(landmark[2], particle_reading[2])

                    prob = np.exp(-(d ** 2) / (const_d)
                                       - (a ** 2) / (const_a))
                    if prob > best_prob:
                        best_prob  = prob
                        best_particle = particle_reading

                if best_particle != None:
                    particle_readings.remove(best_particle)

                prob *= best_prob

        prob *= 0.5 ** abs(len(particle_readings) - len(measured_marker_list))

        return prob

    def resample(particles, weights, grid):

        # first normalize particle weights, grab relevant weight information

        min_weight = 1e-6 # we'll get NaN if the weight is too small
        weights = [min_weight if weight < min_weight else weight for weight in weights]
        normalized_weights = np.divide(weights, np.sum(weights), dtype=np.float64)
        #normalized_weights[np.isnan(normalized_weights)] = 0 # get rid of NaNs

        # next generate new particle distribution based on above probabilities
        particles = np.random.choice(particles, size=setting.PARTICLE_COUNT - int(setting.PARTICLE_COUNT * .03), replace=True, p=normalized_weights)

        return particles

    # now we combine everything
    init_weight = 1e-10
    weights = []

    #some weight that all particles will get if no landmarks found by robot
    if len(measured_marker_list) == 0:
        return particles

    else:
        for particle in particles:

            if (not grid.is_in(particle.x, particle.y)) or (not grid.is_free(particle.x, particle.y)):
                weight = 0 # if not in map or within an obstacle, is a 0
            else:
                particle_readings = particle.read_markers(grid)
                weight = update_weight(measured_marker_list, particle_readings)
            weights.append(weight)

    # resample
    
    measured_particles = resample(particles, weights, grid)

    rand_particle = Particle.create_random(int(setting.PARTICLE_COUNT*.03), grid)

    measured_particles = np.ndarray.tolist(measured_particles) + rand_particle

    return measured_particles


