import asyncio
import concurrent
import numpy as np
import cozmo
from cozmo.util import degrees, distance_mm, speed_mmps
import time
from itertools import chain
import sys
import datetime
import time
import imgclassification_sol
from PIL import Image

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
    cubes = robot.world.wait_until_observe_num_objects(num=1, object_type=cozmo.objects.LightCube, timeout=60)
    look_around.stop()

    if len(cubes) == 1:
        action = robot.pickup_object(cubes[0], num_retries=3)
        action.wait_for_completed()
        print("Completed action: result = %s" % action)
        print("Done.")

def surveillance(robot: cozmo.robot.Robot):
    for i in range(4):
        robot.drive_straight(distance_mm(150), speed_mmps(50)).wait_for_completed()
        robot.turn_in_place(degrees(90)).wait_for_completed()

def fsm(robot: cozmo.robot.Robot):
    img_clf = imgclassification_sol.ImageClassifier()
    (train_raw, train_labels) = img_clf.load_data_from_folder('./train/')

    # convert images into features
    print("before loop ", type(train_raw))
    print("before loop ", train_raw.shape)
    s_row, s_col = train_raw[0].shape[:2]
    print(s_row, " ", s_col)
    train_data = img_clf.extract_image_features(train_raw)

    img_clf.train_classifier(train_data, train_labels)

    robot.enable_device_imu(True, True, True)
    robot.camera.image_stream_enabled = True
    robot.camera.color_image_enabled = False
    robot.camera.enable_auto_exposure()
    robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()

    while True:
        latest_image = robot.world.latest_image
        if(latest_image is None):
            continue
        raw_image = latest_image.raw_image
        np_img = np.array(np.array(raw_image))
        pil_image = Image.fromarray(np_img)
        pil_image = pil_image.resize((320,240))
        pil_image = np.array(pil_image)
        arr = []
        arr.append(pil_image)
        pil_image = np.array(arr)
        features = img_clf.extract_image_features(pil_image)

        predicted_label = img_clf.predict_labels(features)

        if(predicted_label != 'none'):
            robot.say_text(predicted_label[0]).wait_for_completed()
            if(predicted_label == 'order'):
                defuse(robot)

        time.sleep(2)
        print(predicted_label)
def main():
    cozmo.run_program(fsm)

if __name__ == "__main__":
    main()