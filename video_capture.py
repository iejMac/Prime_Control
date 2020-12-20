import cv2


class Camera:
    def __init__(self, cam_num=0):
        # 1 because 0 is front facing laptop camera and 1 is iphone camera
        self.capture_device = cv2.VideoCapture(cam_num)

    def get_frame(self):
        frame = None

        if self.capture_device.isOpened():
            r_val, frame = self.capture_device.read()
        else:
            r_val = False

        return r_val, frame


'''
# import cv2
# import video_capture as vc

cv2.namedWindow("preview")
dev = Camera()

rval, frame = dev.get_frame()

while rval:
    cv2.imshow("preview", frame)

    rval, frame = dev.get_frame()
    print(frame.shape)

    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

# cv2.destroyWindow("preview")
'''


