import cv2
import video_capture as vc
import network_prime

# CAMERA SETUP NOTES:
# =======================
# - Remember to show the network it's training data, so show the right angle and proportions of the street
# - Set up the camera straight


def show_steering(image, angle):
    height, width = image.shape[0:2]
    # Ellipse parameters
    radius = 40
    center = (320, 220)
    axes = (radius, radius)
    startAngle = 180 + 45
    endAngle = 360 - 45
    thickness = 5
    BLACK = 0

    cv2.circle(image, (320, 220), radius=40, color=(255, 255, 255), thickness=4, lineType=8, shift=0)
    cv2.ellipse(image, center, axes, angle, startAngle, endAngle, BLACK, thickness)
    return


def angle_update_rule(current, predicted):
    updated_angle = current*0.7 + predicted*0.3
    return updated_angle


cv2.namedWindow("preview")
dev = vc.Camera(0)
# dev = cv2.VideoCapture('D:/machine_learning_data/SelfDriving/video_tests/video_test1.mp4')

control = network_prime.PrimeControl(brain_path='best_prime_control_139MSE', in_channels=1)
print(control.get_parameter_count())

rval, frame = dev.get_frame()
# dev.set(1, 6000)
# rval, frame = dev.read()

current_steering_angle = 0.0

while rval:

    # Crop out top:
    new_frame = frame[200:, :]
    # Resize:
    new_frame = cv2.resize(new_frame, (160, 70))
    new_frame = cv2.cvtColor(new_frame, cv2.COLOR_RGB2GRAY)
    # show_frame = cv2.resize(frame, (640, 280))
    show_frame = frame

    # Required reshape so neural network can consume the data
    new_frame = new_frame.reshape(-1, 1, 70, 160)
    new_frame = (new_frame/127.5) - 1.0

    y = control.predict(new_frame)[0].item()
    current_steering_angle = angle_update_rule(current_steering_angle, y)
    print(current_steering_angle)

    # show_frame parameter just for imshow function because it makes the window size same as the
    # image size making it very small but we use 70x160 for prediction
    show_steering(show_frame, -current_steering_angle)
    cv2.imshow("preview", show_frame)

    rval, frame = dev.get_frame()
    # rval, frame = dev.read()

    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

cv2.destroyWindow("preview")
