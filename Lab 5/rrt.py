import cozmo
import math
import sys
import time
import random

from cmap import *
from gui import *
from utils import *

MAX_NODES = 20000
goal_center_found = False
#BRIAN ZHU AND MIGUEL GARCIA
def step_from_to(node0, node1, limit=75):
    ########################################################################
    # TODO: please enter your code below.
    # 1. If distance between two nodes is less than limit, return node1
    dist = get_dist(node0, node1)
    if dist < limit:
        return node1
    # 2. Otherwise, return a node in the direction from node0 to node1 whose
    #    distance to node0 is limit. Recall that each iteration we can move
    #    limit units at most
    # 3. Hint: please consider using np.arctan2 function to get vector angle
    # 4. Note: remember always return a Node object
    else:

        # Instead of using theta and calculating x and y like below:
        # -----------------------------------------------------------
        #theta = np.arctan2((node1.y-node0.y), (node1.x-node0.x))
        #xCoord = limit * np.cos(theta)
        #yCoord = limit * np.sin(theta)
        #node = Node((xCoord, yCoord))
        # -------------------------------------------------------------

        # We know we can only go a certain % of the distance
        ratio = limit / dist

        # the difference b/t node0 and node1, times that ratio, (how far we can travel) actually is the offset from node0
        offsetX = node1.x * ratio - node0.x * ratio
        offsetY = node1.y * ratio - node0.y * ratio

        node = Node((
            node0.x + offsetX,
            node0.y + offsetY
        ))

        return node
    
        ############################################################################

    
    
    


def node_generator(cmap):
    rand_node = None
    ############################################################################
    # TODO: please enter your code below.
    # 1. Use CozMap width and height to get a uniformly distributed random node
    # 2. Use CozMap.is_inbound and CozMap.is_inside_obstacles to determine the
    #    legitimacy of the random node.
    # 3. Note: remember always return a Node object

    # first, 5% chance that the goal location is returned
    if random.random() < 0.05:
        goal = cmap.get_goals()[0]
        return Node((goal.x, goal.y))

    # first check if legitimate location (if not, create new Node (and if that's not, continue...))
    rand_node = Node((random.random() * cmap.width, random.random() * cmap.height))

    inside_obstacle = cmap.is_inside_obstacles(rand_node)
    outside_map = not cmap.is_inbound(rand_node)
    while rand_node==None or inside_obstacle or outside_map:
        rand_node = Node((random.random() * cmap.width, random.random()*cmap.height))
        outside_map = not cmap.is_inbound(rand_node)
        inside_obstacle = cmap.is_inside_obstacles(rand_node)
    
    return rand_node
    ############################################################################
    


def RRT(cmap, start):
    cmap.add_node(start)
    map_width, map_height = cmap.get_size()
    while (cmap.get_num_nodes() < MAX_NODES):
        ########################################################################
        # TODO: please enter your code below.

        # 1. Use CozMap.get_random_valid_node() to get a random node. This
        #    function will internally call the node_generator above
        rand_node = cmap.get_random_valid_node()

        # 2. Get the nearest node to the random node from RRT
        # do this by grabbing some (other) random node, and comparing all other nodes to it
        # replace comparison node with any "closer" node
        nodes = cmap.get_nodes()
        best_dist = math.inf
        nearest_node = None
        for node in nodes:
            dist = get_dist(rand_node, node)
            if dist != 0 and dist < best_dist:
                best_dist = dist
                nearest_node = node     

        # 3. Limit the distance RRT can move
        limit_node = step_from_to(nearest_node, rand_node)
        # 4. Add one path from nearest node to random node
        # already done?

        ########################################################################
        
        
        time.sleep(0.01)
        cmap.add_path(nearest_node, limit_node)
        if cmap.is_solved():
            break

    path = cmap.get_path()
    smoothed_path = cmap.get_smooth_path()

    if cmap.is_solution_valid():
        print("A valid solution has been found :-) ")
        print("Nodes created: ", cmap.get_num_nodes())
        print("Path length: ", len(path))
        print("Smoothed path length: ", len(smoothed_path))
    else:
        print("Please try again :-(")



async def CozmoPlanning(robot: cozmo.robot.Robot):
    # Allows access to map and stopevent, which can be used to see if the GUI
    # has been closed by checking stopevent.is_set()
    global cmap, stopevent
    ########################################################################
    # TODO: please enter your code below.
    # Description of function provided in instructions. Potential pseudcode is below

    #assume start position is in cmap and was loaded from emptygrid.json as [50, 35] already
    #assume start angle is 0

    #Add final position as goal point to cmap, with final position being defined as a point that is at the center of the arena 
    #you can get map width and map weight from cmap.get_size()
    await robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()
    await robot.set_lift_height(0).wait_for_completed()
    map_width, map_height = cmap.get_size()

    center_spot = Node((13 * 25.4, 9 * 25.4))
    cmap.add_goal(center_spot)

    #reset the current stored paths in cmap
    #call the RRT function using your cmap as input, and RRT will update cmap with a new path to the target from the start position

    cmap.reset_paths()
    RRT(cmap, cmap.get_start())
    print("starting and getting inital path from RRT without any obstacles")

    #marked and update_cmap are both outputted from detect_cube_and_update_cmap(robot, marked, cozmo_pos).
    #and marked is an input to the function, indicating which cubes are already marked
    #So initialize "marked" to be an empty dictionary and "update_cmap" = False
    marked = dict()
    update_cmap = False

    goal_center = False
    curr_angle = 0

    #while the current cosmo position is not at the goal:
    while True:
        # get path from the cmap
        path = cmap.get_smooth_path()

        #break if path is none or empty, indicating no path was found
        if path is None or len(path) == 0:
            break

        (update_cmap, goal_c, mark) = await detect_cube_and_update_cmap(robot, marked,
                                                                        (robot.pose.position.x, robot.pose.position.y))

        print("starting to run through the path")
        curr_node = path[0]
        # Get the next node from the path
        # drive the robot to next node in path.
        # First turn to the appropriate angle, and then move to it
        # you can calculate the angle to turn through a trigonometric function
        count = 0
        for i in range(len(path)):
            # if i == 0:
            #     continue
            #reinitialze where you are on the map
            cmap.set_start(curr_node)
            dist_difference = ((path[i].x - curr_node.x), (path[i].y - curr_node.y))
            diff_angle = np.arctan2(dist_difference[1], dist_difference[0])
            #print("curr angle ", curr_angle)
            #print("diff_angle ", math.degrees(diff_angle))
            angle_to_turn = coordinate_frame_convert(math.degrees(diff_angle), curr_angle)
            #print("angle to turn ", angle_to_turn)
            await robot.turn_in_place(cozmo.util.degrees(angle_to_turn)).wait_for_completed()
            #print("curr angle and angle to turn addition ", curr_angle + angle_to_turn)
            curr_angle = compute_angle(curr_angle + angle_to_turn)
            #print("curr angle after compute angle", curr_angle)
            (update_cmap, goal_c, _) = await detect_cube_and_update_cmap(robot, marked, curr_node)
            #if we detected a cube, indicated by update_cmap, reset the cmap path, recalculate RRT, and get new paths
            if update_cmap:
                print("cube detected, resetting path")
                if not goal_c:
                    goal_center = True
                angle_to_turn = coordinate_frame_convert(0, curr_angle)
                await robot.turn_in_place(cozmo.util.degrees(angle_to_turn)).wait_for_completed()
                await robot.drive_straight(cozmo.util.distance_mm(0), cozmo.util.speed_mmps(0)).wait_for_completed()
                curr_angle = compute_angle(curr_angle + angle_to_turn)
                cmap.reset_paths()
                cmap.set_start(curr_node)
                RRT(cmap, cmap.get_start())
                break

            dist = get_dist(path[i], curr_node)
            await robot.drive_straight(cozmo.util.distance_mm(dist), cozmo.util.speed_mmps(30)).wait_for_completed()
            #put in a little extra sleep time after finishing to detect cubes
            time.sleep(1)
            curr_node = path[i]
            count += 1
        if update_cmap:
            continue

        goal_reached = goal_center

        if goal_reached:
            print("goal reached, end the while loop")
            break
        if count == (len(path)):
            print("went through whole path")
            break
    ########################################################################

def get_global_node(local_angle, local_origin, node):
    """Helper function: Transform the node's position (x,y) from local coordinate frame specified by local_origin and local_angle to global coordinate frame.
                        This function is used in detect_cube_and_update_cmap()
        Arguments:
        local_angle, local_origin -- specify local coordinate frame's origin in global coordinate frame
        local_angle -- a single angle value
        local_origin -- a Node object
        Outputs:
        new_node -- a Node object that decribes the node's position in global coordinate frame
    """
    ########################################################################
    # TODO: please enter your code below.

    #get trig values
    sin = math.sin(local_angle)
    cos = math.cos(local_angle)
    #y_val = node.y * sin + node.y * cos
    y_val = node.x * sin + node.y * cos
    #x_val = node.x * cos - node.x * sin
    x_val = node.x * cos - node.y * sin
    local_x = local_origin.x
    local_y = local_origin.y
    # declare new node
    #temporary code below to be replaced
    new_node = Node((x_val + local_x, y_val + local_y))
    return new_node
    ########################################################################


async def detect_cube_and_update_cmap(robot, marked, cozmo_pos):
    """Helper function used to detect obstacle cubes and the goal cube.
       1. When a valid goal cube is detected, old goals in cmap will be cleared and a new goal corresponding to the approach position of the cube will be added.
       2. Approach position is used because we don't want the robot to drive to the center position of the goal cube.
       3. The center position of the goal cube will be returned as goal_center.

        Arguments:
        robot -- provides the robot's pose in G_Robot
                 robot.pose is the robot's pose in the global coordinate frame that the robot initialized (G_Robot)
                 also provides light cubes
        cozmo_pose -- provides the robot's pose in G_Arena
                 cozmo_pose is the robot's pose in the global coordinate we created (G_Arena)
        marked -- a dictionary of detected and tracked cubes (goal cube not valid will not be added to this list)

        Outputs:
        update_cmap -- when a new obstacle or a new valid goal is detected, update_cmap will set to True
        goal_center -- when a new valid goal is added, the center of the goal cube will be returned
    """
    global cmap

    # Padding of objects and the robot for C-Space
    cube_padding = 40.
    cozmo_padding = 100.

    # Flags
    update_cmap = False
    goal_center = None

    # Time for the robot to detect visible cubes
    time.sleep(1)

    for obj in robot.world.visible_objects:

        if obj.object_id in marked:
            continue

        # Calculate the object pose in G_Arena
        # obj.pose is the object's pose in G_Robot
        # We need the object's pose in G_Arena (object_pos, object_angle)
        dx = obj.pose.position.x - robot.pose.position.x
        dy = obj.pose.position.y - robot.pose.position.y

        object_pos = Node((cozmo_pos.x+dx, cozmo_pos.y+dy))
        object_angle = obj.pose.rotation.angle_z.radians

        # Define an obstacle by its four corners in clockwise order
        obstacle_nodes = []
        obstacle_nodes.append(get_global_node(object_angle, object_pos, Node((cube_padding, cube_padding))))
        obstacle_nodes.append(get_global_node(object_angle, object_pos, Node((cube_padding, -cube_padding))))
        obstacle_nodes.append(get_global_node(object_angle, object_pos, Node((-cube_padding, -cube_padding))))
        obstacle_nodes.append(get_global_node(object_angle, object_pos, Node((-cube_padding, cube_padding))))
        cmap.add_obstacle(obstacle_nodes)
        marked[obj.object_id] = obj
        update_cmap = True

    return update_cmap, goal_center, marked


def compute_angle(angle):
    if angle < 0:
        angle = angle + 360
        return angle
    if angle > 360:
        angle = angle - 360
        return angle
    return angle


def coordinate_frame_convert(angle1, angle2):
    diff = angle1 - angle2
    if diff <= -180:
        while diff <= -180:
            diff += 360
    if diff > 180:
        while diff > 180:
            diff -= 360
    return diff


def get_current_pose(robot):
    start_x, start_y = cmap.get_start()
    curr_x = robot.pose.position.x
    curr_y = robot.pose.position.y

    return Node((start_x + curr_x, start_y + curr_y))

class RobotThread(threading.Thread):
    """Thread to run cozmo code separate from main thread
    """

    def __init__(self):
        threading.Thread.__init__(self, daemon=True)

    def run(self):
        # Please refrain from enabling use_viewer since it uses tk, which must be in main thread
        cozmo.run_program(CozmoPlanning,use_3d_viewer=False, use_viewer=False)
        stopevent.set()


class RRTThread(threading.Thread):
    """Thread to run RRT separate from main thread
    """

    def __init__(self):
        threading.Thread.__init__(self, daemon=True)

    def run(self):
        while not stopevent.is_set():
            RRT(cmap, cmap.get_start())
            time.sleep(100)
            cmap.reset_paths()
        stopevent.set()

if __name__ == '__main__':
    global cmap, stopevent
    stopevent = threading.Event()
    robotFlag = False
    for i in range(0,len(sys.argv)): #reads input whether we are running the robot version or not
        if (sys.argv[i] == "-robot"):
            robotFlag = True
    if (robotFlag):
        #creates cmap based on empty grid json
        #"start": [50, 35],
        #"goals": [] This is empty
        cmap = CozMap("maps/emptygrid.json", node_generator) 
        robot_thread = RobotThread()
        robot_thread.start()
    else:
        cmap = CozMap("maps/map2.json", node_generator)
        sim = RRTThread()
        sim.start()
    visualizer = Visualizer(cmap)
    visualizer.start()
    stopevent.set()
