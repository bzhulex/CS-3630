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
import math

############ LAB 4 IMPORTS #################
from go_to_goal_cozmo import ParticleFilter
from go_to_goal_cozmo import compute_odometry
from go_to_goal_cozmo import marker_processing
#from go_to_goal_cozmo import run -- Create our own run()

############ LAB 5 IMPORTS #################
from cmap import *
from gui import *
from utils import * # has the Node class

from rrt import step_from_to
from rrt import node_generator
from rrt import RRT
from rrt import CozmoPlanning # main
from rrt import get_global_node
from rrt import detect_cube_and_update_cmap
from rrt import compute_angle, get_current_pose

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

############## SETUP STUFF ###################
# tmp cache
prev_pose = cozmo.util.Pose(0, 0, 0, angle_z=cozmo.util.Angle(degrees=0))
picked_up_flag = False

# map and particle filter
Map_filename = "map_arena.json"
grid = CozGrid(Map_filename)
gui = GUIWindow(grid, show_camera=True)
pf = ParticleFilter(grid)

################ STAGE 1 #####################
'''
Starting Conditions: The robot will begin near the center of the arena with a random orientation. 
It will first have to localize itself, then proceed to the pickup zone. 
Once in the pickup zone, the robot must indicate through an audio cue that it is ready to begin delivery. 
From that point on, a cube will be placed within the pickup area for the robot to deliver.
'''
global cmap, stopevent, curr_angle
#map_width, map_height = cmap.get_size()


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

    dx, dy = rotate_point(curr_x - last_x, curr_y - last_y, -last_h)
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
    marker_list = [(x / grid.scale, y / grid.scale, h) for x, y, h in marker_list]

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


async def run_pf(robot: cozmo.robot.Robot):
    global picked_up_flag, prev_pose, curr_angle
    global grid, gui, pf

    goal = (1.0,  11.75, 0)
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
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float)

    ###################

    # YOUR CODE HERE
    start_time = time.time()
    has_converged = False
    converged_score = 0
    arrived_to_goal = False
    print("start pf loop")
    while True:
        #print("in pf loop")
        if arrived_to_goal and not robot.is_picked_up:
            await robot.drive_wheels(0.0, 0, 0)

        # kidnapped
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

        # INFORMATION UPDATE
        curr_pose = robot.pose
        odom = compute_odometry(curr_pose, cvt_inch=True)
        prev_pose = robot.pose
        marker_list, annotated_image = await marker_processing(robot, camera_settings, show_diagnostic_image=False)
        # PF
        (mean_x, mean_y, mean_h, mean_confidence) = pf.update(odom, marker_list)
        #print("done with pf update")
        #print("gui 0")
        gui.show_particles(pf.particles)
        #print("gui 1")
        gui.show_mean(mean_x, mean_y, mean_h)
        #print("gui 2")
        gui.show_camera_image(annotated_image)
        #print("gui 3")
        gui.updated.set()
        #print("done with gui")
        # pf returns convergence
        if mean_confidence:
            converged_score += 2.5

        # converged and confident of it
        if converged_score >= 10:
            has_converged = True

        # if pf converged but then diverged again
        if has_converged and not mean_confidence:
            converged_score -= 1.5

        # if the pf diverges a lot, then reset
        if converged_score < 0:
            has_converged = False
            converged_score = 0

        temp_score = 1 + converged_score / 10
        #print("start drive wheel")
        await robot.drive_wheels(12.0 / temp_score, -12.0 / temp_score)

        if has_converged:
            await robot.drive_wheels(0.0, 0, 0)
            goal_x = goal[0]
            goal_y = goal[1]
            goal_h = goal[2]

            d_x = goal_x - mean_x
            d_y = goal_y - mean_y

            target = math.atan2(d_y, d_x) * 180.0 / 3.14159

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
            return
        else:
            # if there is no convergence, keep turning in place
            if ((time.time() - start_time) // 1) % 8 < 3 or len(marker_list) <= 0:
                await robot.drive_wheels(12.0, -12, 0)
            elif len(marker_list) > 0:
                markers_loc = marker_list[0][0]
                if markers_loc > 12:
                    await robot.drive_wheels(15.0, 15, 0)
                if markers_loc < 8:
                    await robot.drive_wheels(-15.0, -15, 0)
            else:
                await robot.drive_wheels(0.0, 0, 0)
        #print("end of pf loop")
    ###################

async def go_to_zone(robot: cozmo.robot.Robot, zone, obstacle):
    global cmap, stopevent, curr_angle
    # Clear old goal
    cmap.clear_goals()

    # If obstacle (i.e. every run except for the first one), make sure Fragile Zone is in cmap
    if len(obstacle) > 1:
        # This is ugly, but obstacle has a length of 1 if "None". Else its a list. Could also check if list but doesn't matter i dont think
        cmap.add_obstacle(obstacle)
        print("first run add fragile zone")
        

    # Add zone as a goal.
    cmap.add_goal(zone)

    # Reset paths
    cmap.reset_paths()
    RRT(cmap, cmap.get_start())

    # Go to zone
    while True:
        # get path from the cmap
        path = cmap.get_smooth_path()

        #break if path is none or empty, indicating no path was found
        if path is None or len(path) == 0:
            break

        curr_node = path[0]

        count = 1
        for i in range(len(path)):
            if i == 0:
                continue

            #reinitialze where you are on the map
            cmap.set_start(curr_node)

            # compute turn
            dist_difference = ((path[i].x - curr_node.x), (path[i].y - curr_node.y))
            diff_angle = np.arctan2(dist_difference[1], dist_difference[0])
            print(curr_angle)
            print(type(curr_angle))
            print(math.degrees(curr_angle))
            angle_to_turn = (math.degrees(diff_angle) -  curr_angle)
            await robot.turn_in_place(cozmo.util.degrees(angle_to_turn)).wait_for_completed()

            # update curr_angle
            curr_angle = curr_angle + angle_to_turn

            #print("curr angle after compute angle", curr_angle)
            dist = get_dist(path[i], curr_node)
            await robot.drive_straight(cozmo.util.distance_mm(dist), cozmo.util.speed_mmps(30)).wait_for_completed()
            
            curr_node = path[i]
            count += 1

        # goal_reached = goal_center
        #
        # if goal_reached:
        #     print("goal reached, end the while loop")
        #     break

        if count == (len(path)):
            print("went through whole path")
            break

    return curr_angle # keep track of this globally

async def run(robot: cozmo.robot.Robot):
    global cmap, stopevent, curr_angle
    '''
    Main driver code function.
    Consists of 2 sections:

        - Startup, where we localize the robot and set everything up
        
        - Delivery Loop
            1. Pickup -- detect and pickup the cube
            2. Deliver -- Take the cube and move it to the storage zone, avoiding
            the fragile zone.
            3. Return to pickup -- Once the cube has been placed,
            return to the pickup zone
    '''

    # Robot starts normal
    await robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()
    await robot.set_lift_height(0).wait_for_completed()

    # hard-code zones as goals
    pickup_zone = Node((4.25 * 25.4, 10.5 * 25.4)) # pickup zone is 8.5 x 8.5", top left corner, so just hardcode ~middle
    storage_zone = Node((21.75 * 25.4, 10.5 * 25.4)) # pickup zone is 8.5 x 8.5", top right corner, so just hardcode ~middle
    
    # fragile zone we actually need to indicate the four corners as nodes
    # fragile_corner1 = Node((11.5 * 25.4, 6 * 25.4))
    # fragile_corner2 = Node((11.5 * 25.4, 26 * 25.4))
    # fragile_corner3 = Node((14.5 * 25.4, 26 * 25.4))
    # fragile_corner4 = Node((14.5 * 25.4, 6 * 25.4))


    fragile_corner1 = Node((10.5 * 25.4, 5 * 25.4))
    fragile_corner2 = Node((10.5 * 25.4, 26 * 25.4))
    fragile_corner3 = Node((15 * 25.4, 26 * 25.4))
    fragile_corner4 = Node((15 * 25.4, 5 * 25.4))

    fragile_zone = [fragile_corner1, fragile_corner2, fragile_corner3, fragile_corner4]
    
    print("starting and getting inital path from RRT without any obstacles")

    # initialize helper values/structures
    marked = dict()    #marked (input to cmap funcs) and update_cmap are both outputted from detect_cube_and_update_cmap
    #update_cmap = False
    goal_center = False # used to see if goal_reached
    # Note: probably won't need these variables besides goal_center
    curr_angle = 0 # global angle to save time with calculating rotations


    async def startup():
        global cmap, stopevent
        '''
        During startup, don't need to worry about the fragile zone.
        So we can simply add pickup zone as a goal, go there, clear goals and reset paths.
        Then indicate ready to pickup and deliver with audio cue.
        '''
        print("starting to run pf")
        await run_pf(robot)
        print("done with pf")
        # Add pickup zone as goal, reset paths, make new path to pickup zone, go there
        #curr_angle = go_to_zone(robot, pickup_zone, cmap, obstacle=None, curr_angle = curr_angle)
        curr_angle = 0

        # Indicate that is ready for pickup.
        text = '''
        Ready for pickup!
        '''
        await robot.say_text(text).wait_for_completed()

        return curr_angle

    async def pickup():
        global cmap, stopevent
        '''
        Detect & pickup cube.
        '''
        # Pickup cube.
        # look around and try to find a cube
        print("starting pickup")
        await robot.set_head_angle(cozmo.util.degrees(-20)).wait_for_completed()
        await robot.set_lift_height(0).wait_for_completed()
        # time.sleep(1.5)
        # look_around = robot.start_behavior(cozmo.behavior.BehaviorTypes.LookAroundInPlace).wait_for_completed()
        # try:
        #     cube = robot.world.wait_for_observed_light_cube(timeout=30)
        #     print("Found cube: %s" % cube)
        # except asyncio.TimeoutError:
        #     print("Didn't find a cube")
        # finally:
        #     # whether we find it or not, we want to stop the behavior
        #     look_around.stop()
        #
        # if cube:
        #     robot.stop_all_motors()
        #     action = robot.pickup_object(cube, num_retries=3)
        #     await action.wait_for_completed()
        print("Cozmo is waiting until he sees a cube.")

        cube = await robot.world.wait_for_observed_light_cube()

        print("Cozmo found a cube, and will now attempt to dock with it:")
        action = robot.pickup_object(cube, num_retries=3)
        await action.wait_for_completed()
        print("result:", action.result)
        curr_angle = robot.pose_angle.degrees

        return curr_angle
    async def deliver(curr_angle):
        global cmap, stopevent
        '''
        Robot navigates to the storage zone, avoiding the fragile zone.
        '''
        # Go to storage zone
        print("starting delivery")
        curr_angle = await go_to_zone(robot, storage_zone, fragile_zone)
        
        # Drop off cube.
        await robot.place_object_on_ground_here()

        return curr_angle

    async def return_to_pickup(curr_angle):
        global cmap, stopevent
        '''
        Robot clears goals and resets paths.
        Robot keeps fragile zone as an obstacle, and adds the pickup zone as a goal.
        Robot returns to the pickup zone, avoiding the fragile zone
        '''

        # Go to pickup.
        curr_angle = await go_to_zone(robot, pickup_zone, fragile_zone)

        return curr_angle

    # Start by localizing self and going to pickup zone
    await startup()

    # Then run the pickup-deliver-return_to_pickup loop
    while True:
        print("curr_angle at start ", curr_angle)
        print("picking up...")
        curr_angle = await pickup()
        time.sleep(1.5)
        print("curr_angle after pickup ", curr_angle)
        print("delivering...")
        curr_angle = await deliver()
        time.sleep(1.5)
        print("curr_angle after delivery ", curr_angle)
        print("returning to pickup...")
        curr_angle = await return_to_pickup()
        time.sleep(1.5)
        print("curr_angle after return to pickup ", curr_angle)


    return


###############################################
#### RUN THE ROBOT SAME WAY AS IN LAB4 ########
###############################################
class CozmoThread(threading.Thread):
    
    def __init__(self):
        threading.Thread.__init__(self, daemon=False)

    def run(self):
        cozmo.robot.Robot.drive_off_charger_on_connect = False  # Cozmo can stay on his charger
        cozmo.run_program(run, use_viewer=False)


if __name__ == '__main__':

    # cozmo thread
    global cmap, stopevent, curr_angle
    cmap = CozMap("maps/emptygrid.json", node_generator)
    cozmo_thread = CozmoThread()
    cozmo_thread.start()

    # init
    gui.show_particles(pf.particles)
    gui.show_mean(0, 0, 0)
    gui.start()