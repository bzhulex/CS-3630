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
    def gaussian_prob_density(robot_reading, particle_reading):
        '''
        Find: d = dist_bt_markers
              a = angle_bt_markers

        Return probability
        P(x) = exp(-(d**2)/(2sigma**2) + (angle**2)/(2sigma**2))
        '''
        # set the 2*sigma^2 constant for distance and angle
        const_d = 2 * setting.MARKER_TRANS_SIGMA**2
        const_a = 2 * setting.MARKER_ROT_SIGMA**2

        # calculate dist and angle
        d = grid_distance(robot_reading[0],robot_reading[1],particle_reading[0],particle_reading[1])
        a = diff_heading_deg(robot_reading[2],particle_reading[2])

        # calculate probability from d and a
        prob = np.exp(-(d**2)/const_d + (a**2)/const_a)
        return prob

    def update_weights(particles,measured_marker_list, grid):
        '''
        Obtain real robot sensor readings, z_r

        For each particle, simulate the particles FOV and obtain list of marker robot WOULD see
        if really were at the pose z_pi

        Compare z_pi and z_r and assign a weight to the particle proportional to how well they match
        Comparison will be on distance and angle.
        Comparison loop:
            𝑝𝑟𝑜𝑏= 1.0; 
            for each landmark 
                𝑑= Euclidean distance to landmark 
                𝑝𝑟𝑜𝑏*= Gaussian probability of obtaining a reading at distance 𝑑 
                for this landmark from this particle 
            return 𝑝𝑟𝑜𝑏
        '''
        # init weights
        init_weight = 0.1 #some weight that all particles will get if no landmarks found by robot
        weights = [init_weight]*len(particles)
    
        if len(measured_marker_list) == 0:
            return weights

        # calculate prob for each particle and store in weights list
        for particle in particles:
            prob = 1 # init as 1
            particle_readings = particle.read_markers(grid)

            if len(particle_readings) == 0:
                weights.append(0) # if particle sees no landmarks, that particle gets a 0

            else: 
                # otherwise calculate the weights
                for landmark in measured_marker_list:
                    for particle_reading in particle_readings:
                        prob*=gaussian_prob_density(landmark, particle_reading) #adjust probability
                weights.append(prob)

        return weights

    def resample(particles, weights,grid):
        '''
        Generate new set of n particles
        - Normalize them (divide each particle weight by sum of weights)
        - Generate new particle distro based on probability equal to above normalized prob
        - Throw out the low ones and replace with random sampling
        - Maintain some small percentage of random samples
        - Throw out all particles and start with uniform distro if all of them are unlikely (but not needed)
        '''
        # first normalize particle weights
        SUM_WEIGHTS = sum(weights)
        normalized_weights = [weight/SUM_WEIGHTS for weight in weights]
        
        # next generate new particle distribution based on above probabilities
        threshold = 0.1 # anything less than this will be eliminated and replaced randomly
        for i in range(len(particles)):
            if normalized_weights[i] < threshold:
                # eliminate very low prob. particle and replace with rand samples
                particles[i] = particles[i].create_random(1,grid)[0] # get x,y,heading=None = random.uniform(0,360)
        
        return particles

    # now we combine everything :)    
    weights = update_weights(particles, measured_marker_list, grid)
    measured_particles = resample(particles, weights, grid)

    return measured_particles


