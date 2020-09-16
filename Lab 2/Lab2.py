import asyncio
import concurrent
import numpy as np
import cozmo
from cozmo.util import degrees, distance_mm, speed_mmps, distance_inches
import time
from itertools import chain
import sys
import datetime
import time
import imgclassification_sol
from PIL import Image
import joblib


try:
    from PIL import Image, ImageDraw
except ImportError:
    sys.exit("Cannot import from PIL: Do `pip3 install --user Pillow` to install")

def defuse(robot: cozmo.robot.Robot):
    # Move lift down and tilt the head up
    robot.move_lift(-3)
    robot.set_head_angle(degrees(0)).wait_for_completed()

    # look around and try to find a cube
    look_around = robot.start_behavior(cozmo.behavior.BehaviorTypes.LookAroundInPlace)
    try:
        cube = robot.world.wait_for_observed_light_cube(timeout=30)
        print("Found cube: %s" % cube)
    except asyncio.TimeoutError:
        print("Didn't find a cube")
    finally:
        # whether we find it or not, we want to stop the behavior
        look_around.stop()

    if cube:
        robot.stop_all_motors()
        action = robot.pickup_object(cube, num_retries=3)
        action.wait_for_completed()

        action = robot.drive_straight(distance_inches(11.2), speed_mmps(40))
        action.wait_for_completed()

        action = robot.place_object_on_ground_here(cube, num_retries=3)
        action.wait_for_completed()

        action = robot.turn_in_place(degrees(180))
        action.wait_for_completed()

        action = robot.drive_straight(distance_inches(15), speed_mmps(40))
        action.wait_for_completed()

        action = robot.turn_in_place(degrees(180))
        action.wait_for_completed()
        print("Completed action: result = %s" % action)
        print("Done.")

def surveillance(robot: cozmo.robot.Robot):
    for i in range(4):
        robot.drive_straight(distance_mm(150), speed_mmps(50)).wait_for_completed()
        robot.turn_in_place(degrees(90)).wait_for_completed()

def heights(robot: cozmo.robot.Robot):
    '''
    Drive robot in 'S' pattern (top to bottom, right to left)
    Display animation on robot's face.

    (Not yet tested)
    '''
    robot.drive_wheels(50,60,duration=1) # drive in straight-ish line to the left
    robot.drive_wheels(30,60,duration=1) # drive in curved arc to the left
    robot.drive_wheels(60,50,duration=3) # drive in straight-ish line to the right
    robot.drive_wheels(60,30,duration=1)  # drive in curved arc to the right
    robot.stop_all_motors()
    robot.play_anim_trigger(
        cozmo.anim.Triggers.CodeLabWin).wait_for_completed()  # play animation

def burn_notice(robot: cozmo.robot.Robot):
    '''
    Drive in a square (20cm x 20cm), slowly/continuously lower and raise the lift (2s),
    and state "I am not a spy". At end, lower the lift and return to idle state.

    (Not yet tested, also add animation)
    '''
    def drive_side():
        robot.move_lift(1.78)
        robot.drive_straight(distance_mm(200), speed_mmps(50), in_parallel=True).wait_for_completed()
        robot.turn_in_place(degrees(90)).wait_for_completed()
        return

    for i in range(4):
        drive_side()

    robot.LiftPosition(ratio = 0)


def fsm(robot: cozmo.robot.Robot):
    img_clf = imgclassification_sol.ImageClassifier()
    model = joblib.load('trained_model.pkl')
    robot.enable_device_imu(True, True, True)
    robot.camera.image_stream_enabled = True
    robot.camera.color_image_enabled = False
    robot.camera.enable_auto_exposure()
    robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()

    while True:
        images = []
        for i in range(10):
            latest_image = robot.world.latest_image
            if (latest_image is None):
                continue
            raw_image = latest_image.raw_image
            np_img = np.array(raw_image)
            pil_image = Image.fromarray(np_img)
            pil_image = pil_image.resize((320, 240))
            pil_image = np.array(pil_image)
            images.append(pil_image)
            time.sleep(.5)

        features = img_clf.extract_image_features(np.array(images))
        predicted_labels = model.predict(features)
        print("10 predictions: ", predicted_labels)
        unique_preds, counts = np.unique(np.array(predicted_labels), return_counts=True)
        prediction = unique_preds[np.argmax(counts)]
        print("prediction: ", prediction)

        if(prediction != 'none'):
            robot.say_text(prediction).wait_for_completed()
            if(prediction == 'order'):
                defuse(robot)
            elif(prediction == 'drone'):
                heights(robot)
            elif(prediction == 'inspection'):
                burn_notice(robot)

        time.sleep(2)

def main():
    cozmo.run_program(fsm)

if __name__ == "__main__":
    main()
