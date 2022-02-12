import cv2
import numpy as np
import time
from sklearn.metrics import pairwise


class HandFingers:
    """
    Detects number of fingers
    """
    def __init__(self, l=500, t=10, r=1100, b=500, acc_wt=0.5, n=60):
        """
        Take location of Region of Interest

        :param l: int, Leftmost point of RoI
        :param t: int, Topmost point of RoI
        :param r: int, Rightmost point of RoI
        :param b: int, Bottommost point of RoI
        :param acc_wt: float, Weight given to input image
        :param n: int, Number of frames to hold detecting object
        """
        self.bg = None
        self.acc_wt = acc_wt
        self.roi_l = l
        self.roi_t = t
        self.roi_r = r
        self.roi_b = b
        self.no_of_frames = n

    def _accumulation(self, frame) -> None:
        """
        Compute accumulation model
        :param frame: ndArray, RoI
        :return: None
        """
        if self.bg is None:
            self.bg = frame.copy().astype("float")
            return None
        # add to accumulation model
        cv2.accumulateWeighted(frame, self.bg, self.acc_wt)

    def _detectChange(self, frame):
        """
        Detects outer structure of hand
        :param frame: ndArray, RoI
        :return:
        """
        # Find the changes
        dst = cv2.absdiff(frame, cv2.convertScaleAbs(self.bg))

        # Threshold, dilate and find contours of hand
        thresh = cv2.threshold(dst, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # find largest contour
        if len(contours) != 0:
            hand_segment = max(contours, key=cv2.contourArea)
            return thresh, hand_segment
        else:
            return None, None

    @staticmethod
    def fingerCounter(thresh, hand_segment):
        """

        :param thresh: ndArray, Threshold img
        :param hand_segment: ndArray, Boundary of Hand
        :return: tuple, (NumberOfFingers, FingerDetectionFrame, FingerLocation)
        """
        count = 0
        convex_hull = cv2.convexHull(hand_segment)

        # four farther points in convex hull
        left = tuple(convex_hull[convex_hull[:, :, 0].argmin()][0])
        top = tuple(convex_hull[convex_hull[:, :, 1].argmin()][0])
        right = tuple(convex_hull[convex_hull[:, :, 0].argmax()][0])
        bottom = tuple(convex_hull[convex_hull[:, :, 1].argmax()][0])

        # center of hand
        c_x = (left[0] + right[0]) // 2
        c_y = (top[1] + bottom[1]) // 2

        # get max distance from center to 4 pts
        max_distance = pairwise.euclidean_distances([(c_x, c_y)], Y=[left, right, top, bottom])[0].max()

        # Detect fingers
        r = int(0.65 * max_distance)
        circular_roi = np.zeros(thresh.shape, dtype="uint8")
        cv2.circle(circular_roi, (c_x, c_y), r, 255, 10)
        circular_roi = cv2.bitwise_and(thresh, thresh, mask=circular_roi)
        contours, _ = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Finger if object is far below from center of
        # circular_roi then object is probably wrist
        finger_tips = []
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            not_wrist = ((c_y * 1.25) > (y + h))
            if not_wrist:
                finger_tips.append((x, y, w, h))
                count += 1

        return count, circular_roi, finger_tips

    def run(self, frame, width=1280, height=720, draw=True, show_thresh=True):
        """
        Driver of HandFingers class
        :param frame: ndArray, Current Frame
        :param width: int, Width of Frame
        :param height: int, Height of Frame
        :param draw: bool, Draw Hand
        :param show_thresh: bool, Show FingerDetectionFrame
        :return: tuple, (ProcessedFrame, Fingers)
        """
        fingers = 0
        frame = cv2.flip(frame, 1)
        roi = frame[self.roi_t:self.roi_b, self.roi_l:self.roi_r]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi = cv2.GaussianBlur(roi, (5, 5), 0)
        while self.no_of_frames >= 0:
            self.no_of_frames -= 1
            self._accumulation(roi)
            cv2.putText(frame, 'Wait!', (int(width * 0.45), int(height * 0.45)),
                        cv2.FONT_HERSHEY_PLAIN, 10, (0, 0, 255), 4, cv2.LINE_AA)

        thresh, hand_segment = self._detectChange(roi)
        # Draw hand
        if hand_segment is not None:
            if draw:
                cv2.drawContours(frame, [hand_segment + (self.roi_l, self.roi_t)], -1,
                                 (255, 0, 0), 2, cv2.LINE_AA)
            fingers, roi_limit, f_tip = self.fingerCounter(thresh, hand_segment)

            # As much as possible: Limit the draws to 5 finger
            for (x, y, w, h) in f_tip:
                # cv2.circle(frame, (x+self.roi_l, y+self.roi_t), 10, (0, 0, 255), -1)
                cv2.rectangle(frame, (x + self.roi_l, y + self.roi_t),
                              (x + self.roi_l + w, y + self.roi_t + h), (0, 0, 255), 2)

            # Display number of fingers
            cv2.putText(frame, f'#Fingers: {fingers}', (int(width * 0.01), int(height * 0.05)),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2, cv2.LINE_AA)
            if show_thresh:
                cv2.imshow("Detection", cv2.bitwise_xor(thresh, roi_limit))

        return frame, fingers


def main():
    width = 1280
    height = 720
    if height < 480:
        l, t, r, b = 300, 20, 500, 200
    else:
        l, t, r, b = 600, 20, 1200, 500
    p_time: float = 0
    obj = HandFingers(l, t, r, b)

    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    while True:
        success, img = cam.read()
        if success:
            img, _ = obj.run(img, width, height)
            # Place hand text
            cv2.rectangle(img, (l-3, t), (l+130, t-25),
                          (0, 255, 0), -1)
            cv2.putText(img, 'Place your hand', (l+5, t-10),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            # hand frame
            cv2.rectangle(img, (l, t), (r, b), (0, 255, 0), 5)
            c_time: float = time.time()
            fps: int = int(1 / (c_time - p_time))
            p_time = c_time
            # Put height and width on img
            cv2.putText(img, f'({width} x {height})', (int(width * 0.85), int(height * 0.98)),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2, cv2.LINE_AA)
            if fps > 20:
                cv2.putText(img, str(fps), (int(width * 0.01), int(height * 0.98)),
                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(img, str(fps), (int(width * 0.01), int(height * 0.98)),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
            cv2.imshow("Camera", img)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
