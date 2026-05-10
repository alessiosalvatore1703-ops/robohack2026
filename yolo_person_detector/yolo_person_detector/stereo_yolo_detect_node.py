"""Simple YOLO detector for the /stereo_person image pipeline."""

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float32
from vision_msgs.msg import (
    BoundingBox2D,
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
    Pose2D,
)

from .image_conversion import (
    bgr8_to_compressed_imgmsg,
    bgr8_to_image_msg,
    image_msg_to_bgr8,
)
from .yolo_wrapper import InferenceResult, YOLOWrapper


SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
    durability=DurabilityPolicy.VOLATILE,
)

RELIABLE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    durability=DurabilityPolicy.VOLATILE,
)


def qos_from_publishers(infos) -> QoSProfile:
    if not infos:
        return SENSOR_QOS
    pub_qos = infos[0].qos_profile
    return QoSProfile(
        reliability=pub_qos.reliability,
        history=HistoryPolicy.KEEP_LAST,
        depth=5,
        durability=pub_qos.durability,
    )


class StereoYoloDetectNode(Node):
    """Run YOLO on /stereo_person/input_image and publish annotated frames."""

    def __init__(self):
        super().__init__("stereo_yolo_detect")

        self.declare_parameter("input_topic", "/stereo_person/input_image")
        self.declare_parameter("detections_topic", "/stereo_person/detections")
        self.declare_parameter("annotated_topic", "/stereo_person/annotated_image")
        self.declare_parameter(
            "annotated_compressed_topic",
            "/stereo_person/annotated_image/compressed",
        )
        self.declare_parameter("inference_time_topic", "/stereo_person/inference_time")
        self.declare_parameter("jpeg_quality", 85)
        self.declare_parameter("model_path", "yolov8n.pt")
        self.declare_parameter("confidence_threshold", 0.5)
        self.declare_parameter("nms_threshold", 0.45)
        self.declare_parameter("device", "cpu")
        self.declare_parameter("input_size", 640)

        input_topic = self.get_parameter("input_topic").value
        detections_topic = self.get_parameter("detections_topic").value
        annotated_topic = self.get_parameter("annotated_topic").value
        annotated_compressed_topic = self.get_parameter(
            "annotated_compressed_topic"
        ).value
        inference_time_topic = self.get_parameter("inference_time_topic").value
        self.jpeg_quality = self._jpeg_quality()

        self.yolo = YOLOWrapper(
            model_path=self.get_parameter("model_path").value,
            confidence_threshold=float(
                self.get_parameter("confidence_threshold").value
            ),
            nms_threshold=float(self.get_parameter("nms_threshold").value),
            device=self.get_parameter("device").value,
            input_size=int(self.get_parameter("input_size").value),
        )

        infos = self.get_publishers_info_by_topic(input_topic)
        self.image_sub = self.create_subscription(
            Image, input_topic, self._image_callback, qos_from_publishers(infos)
        )
        self.detection_pub = self.create_publisher(
            Detection2DArray, detections_topic, RELIABLE_QOS
        )
        self.annotated_pub = self.create_publisher(Image, annotated_topic, SENSOR_QOS)
        self.annotated_compressed_pub = self.create_publisher(
            CompressedImage, annotated_compressed_topic, SENSOR_QOS
        )
        self.inference_pub = self.create_publisher(
            Float32, inference_time_topic, RELIABLE_QOS
        )

        self.frame_count = 0
        self.get_logger().info(
            f"YOLO detector subscribed to {input_topic}; annotated outputs on "
            f"{annotated_topic} and {annotated_compressed_topic}"
        )

    def _image_callback(self, msg: Image) -> None:
        try:
            image = image_msg_to_bgr8(msg)
        except Exception as exc:
            self.get_logger().warn(f"Image conversion failed: {exc}")
            return

        try:
            result = self.yolo.detect(image)
        except Exception as exc:
            self.get_logger().error(f"YOLO inference failed: {exc}")
            return

        self.frame_count += 1
        self._publish_detections(result, msg.header)
        self._publish_inference_time(result.inference_time_ms)
        self._publish_annotated(image, result, msg.header)

        if self.frame_count % 30 == 0:
            self.get_logger().info(
                f"Frame {self.frame_count}: {len(result.detections)} person(s), "
                f"{result.inference_time_ms:.1f} ms"
            )

    def _publish_detections(self, result: InferenceResult, header) -> None:
        array = Detection2DArray()
        array.header = header

        for det in result.detections:
            detection = Detection2D()
            detection.header = header

            bbox = BoundingBox2D()
            bbox.center = Pose2D()
            bbox.center.position.x = float(det.bbox_x + det.bbox_w / 2.0)
            bbox.center.position.y = float(det.bbox_y + det.bbox_h / 2.0)
            bbox.center.theta = 0.0
            bbox.size_x = float(det.bbox_w)
            bbox.size_y = float(det.bbox_h)
            detection.bbox = bbox

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(det.class_id)
            hyp.hypothesis.score = float(det.confidence)
            detection.results.append(hyp)

            array.detections.append(detection)

        self.detection_pub.publish(array)

    def _publish_inference_time(self, inference_time_ms: float) -> None:
        msg = Float32()
        msg.data = float(inference_time_ms)
        self.inference_pub.publish(msg)

    def _publish_annotated(self, image, result: InferenceResult, header) -> None:
        annotated = image.copy()
        for det in result.detections:
            x1 = det.bbox_x
            y1 = det.bbox_y
            x2 = det.bbox_x + det.bbox_w
            y2 = det.bbox_y + det.bbox_h
            color = (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"Person {det.confidence:.2f}"
            label_size, _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
            )
            label_y = max(label_size[1] + 6, y1)
            cv2.rectangle(
                annotated,
                (x1, label_y - label_size[1] - 6),
                (x1 + label_size[0] + 6, label_y),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 3, label_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                1,
            )

        stats = f"{result.inference_time_ms:.1f} ms | {len(result.detections)} person(s)"
        cv2.putText(
            annotated,
            stats,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        out = bgr8_to_image_msg(annotated, header)
        self.annotated_pub.publish(out)

        compressed = self._cv2_to_compressed_imgmsg(annotated, header)
        if compressed is not None:
            self.annotated_compressed_pub.publish(compressed)

    def _cv2_to_compressed_imgmsg(self, image, header):
        msg = bgr8_to_compressed_imgmsg(
            image, header=header, jpeg_quality=self.jpeg_quality
        )
        if msg is None:
            self.get_logger().warn("Failed to JPEG-encode annotated image")
        return msg

    def _jpeg_quality(self) -> int:
        quality = int(self.get_parameter("jpeg_quality").value)
        return min(max(quality, 1), 100)


def main(args=None):
    rclpy.init(args=args)
    node = StereoYoloDetectNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
