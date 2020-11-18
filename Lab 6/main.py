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
global cmap, stopevent
map_width, map_height = cmap.get_size()

async def go_to_zone(robot: cozmo.robot.Robot, zone, cmap, obstacle=None, curr_angle = 0):

    # Clear old goal
    cmap.clear_goals()

    # If obstacle (i.e. every run except for the first one), make sure Fragile Zone is in cmap
    if len(obstacle) > 1:
        # This is ugly, but obstacle has a length of 1 if "None". Else its a list. Could also check if list but doesn't matter i dont think
        cmap.add_obstacle(obstacle)

        

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
            angle_to_turn = (math.degrees(diff_angle) -  curr_angle)
            await robot.turn_in_place(cozmo.util.degrees(angle_to_turn)).wait_for_completed()

            # update curr_angle
            curr_angle = curr_angle + angle_to_turn

            #print("curr angle after compute angle", curr_angle)
            dist = get_dist(path[i], curr_node)
            await robot.drive_straight(cozmo.util.distance_mm(dist), cozmo.util.speed_mmps(30)).wait_for_completed()
            
            curr_node = path[i]
            count += 1

        goal_reached = goal_center

        if goal_reached:
            print("goal reached, end the while loop")
            break

        if count == (len(path)):
            print("went through whole path")
            break

    return curr_angle # keep track of this globally

async def run(robot: cozmo.robot.Robot):
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
    pickup_zone = Node((5.25 * 25.4, 5 * 25.4)) # pickup zone is 8.5 x 8.5", top left corner, so just hardcode ~middle
    storage_zone = Node((20.25 * 25.4, 5 * 25.4)) # pickup zone is 8.5 x 8.5", top right corner, so just hardcode ~middle
    
    # fragile zone we actually need to indicate the four corners as nodes
    fragile_corner1 = Node((11.5 * 25.4, 0 * 25.4))
    fragile_corner2 = Node((11.5 * 25.4, 0 * 25.4))
    fragile_corner3 = Node((14.5 * 25.4, 12 * 25.4))
    fragile_corner4 = Node((14.5 * 25.4, 12 * 25.4))
    fragile_zone = [fragile_corner1, fragile_corner2, fragile_corner3, fragile_corner4]
    
    print("starting and getting inital path from RRT without any obstacles")

    # initialize helper values/structures
    marked = dict()    #marked (input to cmap funcs) and update_cmap are both outputted from detect_cube_and_update_cmap
    #update_cmap = False
    goal_center = False # used to see if goal_reached
    # Note: probably won't need these variables besides goal_center
    curr_angle = 0 # global angle to save time with calculating rotations


    async def startup():
        '''
        During startup, don't need to worry about the fragile zone.
        So we can simply add pickup zone as a goal, go there, clear goals and reset paths.
        Then indicate ready to pickup and deliver with audio cue.
        '''

        # Add pickup zone as goal, reset paths, make new path to pickup zone, go there
        curr_angle = go_to_zone(robot, pickup_zone, cmap, obstacle=None, curr_angle = curr_angle):


        # Indicate that is ready for pickup.
        text = '''
        Ready for pickup!
        '''
        robot.say_text(text).wait_for_completed()

        return curr_angle

     async def pickup():
        '''
        Detect & pickup cube.
        '''
        # Pickup cube.

        return curr_angle

    async def deliver():
        '''
        Robot navigates to the storage zone, avoiding the fragile zone.
        '''
        # Go to storage zone
        curr_angle = go_to_zone(robot, storage_zone, cmap, obstacle=fragile_zone, curr_angle)
        
        # Drop off cube.


        return curr_angle

    async def return_to_pickup():
        '''
        Robot clears goals and resets paths.
        Robot keeps fragile zone as an obstacle, and adds the pickup zone as a goal.
        Robot returns to the pickup zone, avoiding the fragile zone
        '''

        # Go to pickup.
        curr_angle = go_to_zone(robot, pickup_zone, cmap, obstacle=fragile_zone, curr_angle)

        return curr_angle

    # Start by localizing self and going to pickup zone
    curr_angle = startup()

    # Then run the pickup-deliver-return_to_pickup loop
    while True:
        print("picking up...")
        curr_angle = pickup()
        time.sleep(500)
        print("deliverying...")
        curr_angle = deliver()
        time.sleep(500)
        print("returning to pickup...")
        curr_angle = return_to_pickup()
        time.sleep(500)

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
    cozmo_thread = CozmoThread()
    cozmo_thread.start()

    # init
    gui.show_particles(pf.particles)
    gui.show_mean(0, 0, 0)
    gui.start()