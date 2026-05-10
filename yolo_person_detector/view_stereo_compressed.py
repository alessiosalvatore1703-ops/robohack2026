import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import cv2
import numpy as np
import threading

LEFT_TOPIC = "/aima/hal/sensor/stereo_head_front_left/rgb_image/compressed"
RIGHT_TOPIC = "/aima/hal/sensor/stereo_head_front_right/rgb_image/compressed"

class RealtimeStereoViewer(Node):
    def __init__(self):
        super().__init__("realtime_stereo_viewer")

        # Keep only the newest frame, do not build a queue.
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.left_msg = None
        self.right_msg = None
        self.lock = threading.Lock()

        self.left_sub = self.create_subscription(
            CompressedImage,
            LEFT_TOPIC,
            self.left_callback,
            qos
        )

        self.right_sub = self.create_subscription(
            CompressedImage,
            RIGHT_TOPIC,
            self.right_callback,
            qos
        )

        self.get_logger().info("Realtime stereo viewer started")
        self.get_logger().info(f"Left:  {LEFT_TOPIC}")
        self.get_logger().info(f"Right: {RIGHT_TOPIC}")

        # Display at about 20 Hz max.
        self.timer = self.create_timer(0.05, self.display_latest)

    def left_callback(self, msg):
        with self.lock:
            self.left_msg = msg

    def right_callback(self, msg):
        with self.lock:
            self.right_msg = msg

    def decode(self, msg):
        arr = np.frombuffer(msg.data, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def display_latest(self):
        with self.lock:
            left_msg = self.left_msg
            right_msg = self.right_msg

        if left_msg is None or right_msg is None:
            return

        left = self.decode(left_msg)
        right = self.decode(right_msg)

        if left is None or right is None:
            return

        # Optional: resize to reduce display load.
        scale = 0.25
        left = cv2.resize(left, None, fx=scale, fy=scale)
        right = cv2.resize(right, None, fx=scale, fy=scale)

        if left.shape[:2] != right.shape[:2]:
            right = cv2.resize(right, (left.shape[1], left.shape[0]))

        combined = np.hstack((left, right))

        cv2.putText(
            combined, "LEFT", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2
        )
        cv2.putText(
            combined, "RIGHT", (left.shape[1] + 20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2
        )

        cv2.imshow("Realtime stereo: left | right", combined)
        cv2.waitKey(1)

def main():
    rclpy.init()
    node = RealtimeStereoViewer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()