## If you run into an "[NSApplication _setup] unrecognized selector" problem on macOS,
## try uncommenting the following snippet

# try:
#     import matplotlib
#     matplotlib.use('TkAgg')
# except ImportError:
#     pass

#BRIAN ZHU AND MIGUEL GARCIA

from skimage import color
import cozmo
import numpy as np
from numpy.linalg import inv
import threading
import time
import sys
import asyncio
from PIL import Image

from markers import detect, annotator

from grid import CozGrid
from gui import GUIWindow
from particle import Particle, Robot
from setting import *
from particle_filter import *
from utils import *
from cozmo.util import degrees, distance_mm, speed_mmps, Pose, Angle
from math import atan2

#particle filter functionality
class ParticleFilter:

    def __init__(self, grid):
        self.particles = Particle.create_random(PARTICLE_COUNT, grid)
        self.grid = grid

    def update(self, odom, r_marker_list):

        # ---------- Motion model update ----------
        self.particles = motion_update(self.particles, odom)

        # ---------- Sensor (markers) model update ----------
        self.particles = measurement_update(self.particles, r_marker_list, self.grid)

        # ---------- Show current state ----------
        # Try to find current best estimate for display
        m_x, m_y, m_h, m_confident = compute_mean_pose(self.particles)
        return (m_x, m_y, m_h, m_confident)

# tmp cache
prev_pose = cozmo.util.Pose(0, 0, 0, angle_z=cozmo.util.Angle(degrees=0))
picked_up_flag = False

# goal location for the robot to drive to, (x, y, theta)
goal = (6,10,0)

# map
Map_filename = "map_arena.json"
grid = CozGrid(Map_filename)
gui = GUIWindow(grid, show_camera=True)
pf = ParticleFilter(grid)

def compute_odometry(curr_pose, cvt_inch=True):
    '''
    Compute the odometry given the current pose of the robot (use robot.pose)

    Input:
        - curr_pose: a cozmo.robot.Pose representing the robot's current location
        - cvt_inch: converts the odometry into grid units
    Returns:
        - 3-tuple (dx, dy, dh) representing the odometry
    '''

    global prev_pose, picked_up_flag
    last_x, last_y, last_h = prev_pose.position.x, prev_pose.position.y, \
                             prev_pose.rotation.angle_z.degrees
    curr_x, curr_y, curr_h = curr_pose.position.x, curr_pose.position.y, \
        curr_pose.rotation.angle_z.degrees
    
    dx, dy = rotate_point(curr_x-last_x, curr_y-last_y, -last_h)
    if cvt_inch:
        dx, dy = dx / grid.scale, dy / grid.scale

    return (dx, dy, diff_heading_deg(curr_h, last_h))


async def marker_processing(robot, camera_settings, show_diagnostic_image=False):
    '''
    Obtain the visible markers from the current frame from Cozmo's camera. 
    Since this is an async function, it must be called using await, for example:

        markers, camera_image = await marker_processing(robot, camera_settings, show_diagnostic_image=False)

    Input:
        - robot: cozmo.robot.Robot object
        - camera_settings: 3x3 matrix representing the camera calibration settings
        - show_diagnostic_image: if True, shows what the marker detector sees after processing
    Returns:
        - a list of detected markers, each being a 3-tuple (rx, ry, rh) 
          (as expected by the particle filter's measurement update)
        - a PIL Image of what Cozmo's camera sees with marker annotations
    '''

    global grid

    # Wait for the latest image from Cozmo
    image_event = await robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage, timeout=30)

    # Convert the image to grayscale
    image = np.array(image_event.image)
    image = color.rgb2gray(image)
    
    # Detect the markers
    markers, diag = detect.detect_markers(image, camera_settings, include_diagnostics=True)

    # Measured marker list for the particle filter, scaled by the grid scale
    marker_list = [marker['xyh'] for marker in markers]
    marker_list = [(x/grid.scale, y/grid.scale, h) for x,y,h in marker_list]

    # Annotate the camera image with the markers
    if not show_diagnostic_image:
        annotated_image = image_event.image.resize((image.shape[1] * 2, image.shape[0] * 2))
        annotator.annotate_markers(annotated_image, markers, scale=2)
    else:
        diag_image = color.gray2rgb(diag['filtered_image'])
        diag_image = Image.fromarray(np.uint8(diag_image * 255)).resize((image.shape[1] * 2, image.shape[0] * 2))
        annotator.annotate_markers(diag_image, markers, scale=2)
        annotated_image = diag_image

    return marker_list, annotated_image


async def run(robot: cozmo.robot.Robot):
    global picked_up_flag, prev_pose
    global grid, gui, pf

    # start streaming
    robot.camera.image_stream_enabled = True
    robot.camera.color_image_enabled = False
    robot.camera.enable_auto_exposure()
    await robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()
    await robot.set_lift_height(0).wait_for_completed()

    # Obtain the camera intrinsics matrix
    fx, fy = robot.camera.config.focal_length.x_y
    cx, cy = robot.camera.config.center.x_y
    camera_settings = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=np.float)
            
    ###################

    # YOUR CODE HERE
    start_time = time.time()
    has_converged = False
    converged_score = 0
    arrived_to_goal = False

    while True:

        if arrived_to_goal and not robot.is_picked_up:
            await robot.drive_wheels(0.0, 0, 0)

        #kidnapped
        if robot.is_picked_up:
            picked_up_flag = False
            arrived_to_goal = False
            has_converged = False
            converged_score = 0
            await robot.drive_wheels(0.0, 0, 0)
            await robot.play_anim_trigger(cozmo.anim.Triggers.CodeLabDejected).wait_for_completed()

            while robot.is_picked_up:
                await robot.drive_wheels(0.0, 0, 0)

            prev_pose = cozmo.util.Pose(0, 0, 0, angle_z=cozmo.util.Angle(degrees=0))
            pf.particles = Particle.create_random(PARTICLE_COUNT, grid)

            continue


        #INFORMATION UPDATE
        curr_pose = robot.pose
        odom = compute_odometry(curr_pose, cvt_inch=True)
        prev_pose = robot.pose
        marker_list, annotated_image = await marker_processing(robot, camera_settings, show_diagnostic_image=False)

        #PF
        (mean_x, mean_y, mean_h, mean_confidence) = pf.update(odom, marker_list)

        gui.show_particles(pf.particles)
        gui.show_mean(mean_x, mean_y, mean_h)
        gui.show_camera_image(annotated_image)
        gui.updated.set()

        #pf returns convergence
        if mean_confidence:
            converged_score += 2.5

        # converged and confident of it
        if converged_score >= 25:
            has_converged = True

        #if pf converged but then diverged again
        if has_converged and not mean_confidence:
            converged_score -= 1.5

        # if the pf diverges a lot, then reset
        if converged_score < 0:
            has_converged = False
            converged_score = 0

        temp_score = 1 + converged_score / 10
        await robot.drive_wheels(15.0 / temp_score, -15.0 / temp_score)

        if has_converged:
            await robot.drive_wheels(0.0, 0, 0)
            goal_x = goal[0]
            goal_y = goal[1]
            goal_h = goal[2]

            d_x = goal_x - mean_x
            d_y = goal_y - mean_y

            target = atan2(d_y, d_x) * 180.0 / 3.14159

            initial_degree = diff_heading_deg(target, mean_h)
            zero_degree = diff_heading_deg(goal_h, target)
            distance = grid_distance(mean_x, mean_y, goal_x, goal_y)

            # turn to the right angle
            await robot.turn_in_place(degrees(int(initial_degree * 0.95))).wait_for_completed()
            # drive to the goal
            await robot.drive_straight(distance_mm(distance * grid.scale * 0.95), speed_mmps(50)).wait_for_completed()
            # turn to zero degrees
            await robot.turn_in_place(degrees(int(zero_degree * 0.975))).wait_for_completed()
            # be happy!
            await robot.play_anim_trigger(cozmo.anim.Triggers.CodeLabHappy).wait_for_completed()

            arrived_to_goal = True
        else:
            # if there is no convergence, keep turning in place
            if ((time.time() - start_time) // 1) % 8 < 3 or len(marker_list) <= 0:
                await robot.drive_wheels(12.0, -12, 0)
            elif len(marker_list) > 0:
                markers_loc = marker_list[0][0]
                if markers_loc > 12:
                    await robot.drive_wheels(35.0, 35, 0)
                if markers_loc < 8:
                    await robot.drive_wheels(-35.0, -35, 0)
            else:
                await robot.drive_wheels(0.0, 0, 0)

    ###################

class CozmoThread(threading.Thread):
    
    def __init__(self):
        threading.Thread.__init__(self, daemon=False)

    def run(self):
        cozmo.robot.Robot.drive_off_charger_on_connect = False  # Cozmo can stay on his charger
        cozmo.run_program(run, use_viewer=False)


if __name__ == '__main__':

    # cozmo thread
    cozmo_thread = CozmoThread()
    cozmo_thread.start()

    # init
    gui.show_particles(pf.particles)
    gui.show_mean(0, 0, 0)
    gui.start()

