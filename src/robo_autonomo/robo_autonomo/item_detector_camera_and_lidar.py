import rclpy
import cv2
import numpy as np
import tf2_ros
import geometry_msgs.msg
import math
import atexit
import inference
import tf2_geometry_msgs
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, CameraInfo
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import PointStamped

qos_profile = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=10
)

class ObjectDetector(Node):
    def __init__(self):
        super().__init__('object_detector')
        self.declare_parameter('debug', False)

        self.debug = self.get_parameter('debug').value

        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_profile)
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera_info',
            self.camera_info_callback,
            10)
        self.broadcasting_timer = self.create_timer(1.0, self.broadcast_all_tf)
        self.detection_timer = self.create_timer(1.0, self.detect_items)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
    
        self.bridge = CvBridge()
        self.latest_scan = None
        self.latest_image = None
        self.camera_info = None
        self.detections = []
        self.model = inference.get_model("my-first-project-j6luy/2")

        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        if self.debug:
            self.debug_pub = self.create_publisher(Image, '/debug_item_detection', 10)
            self.get_logger().info("Debug mode enabled; publishing to /debug_item_detection.")

        # Ensure shutdown_callback gets called on exit
        atexit.register(self.shutdown_callback)

    def camera_info_callback(self, msg: CameraInfo):
        self.camera_info = msg

    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg

    def image_callback(self, msg: Image):
        self.latest_image = msg

    def broadcast_all_tf(self):
        for index, detection in enumerate(self.detections):
            self.broadcast_tf(detection['position'], f"{detection['label']}_{index}")

    def detect_items(self):
        if self.camera_info is None:
            self.get_logger().warn("No camera info received yet. Skipping image processing.")
            return
        
        if self.latest_image is None:
            self.get_logger().warn("No image received yet. Skipping image processing.")
            return

        if self.latest_scan is None:
            self.get_logger().warn("No lidar scan received yet.")
            return None

        try:
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error("CV Bridge error: {}".format(e))
            return

        debug_image = cv_image.copy() if self.debug else None

        tmp_file = '/tmp/object_detection.jpg'
        cv2.imwrite(tmp_file, cv_image)
        
        detections = self.model.infer(image=tmp_file)

        self.get_logger().debug(f"detections: {detections}")
        for det in detections:
            for pred in det.predictions:
                if pred.confidence < 0.6:
                    continue
                pred_id = self.compute_new_detection(pred.x, pred.class_name, debug_image)
                if debug_image is not None:
                    cv2.rectangle(debug_image, (int(pred.x - pred.width / 2), int(pred.y - pred.height / 2)), (int(pred.x + pred.width / 2), int(pred.y + pred.height / 2)), (0, 255, 0), 2)
                    cv2.putText(debug_image, f"{pred_id} ({pred.confidence:.2f})", (int(pred.x - pred.width / 2), int(pred.y - pred.height / 2 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if debug_image is not None:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding='bgr8')
            self.debug_pub.publish(debug_msg)


    def compute_new_detection(self, x, label, debug_image=None):
        fx = self.camera_info.k[0]
        cx = self.camera_info.k[2]

        angle_offset = math.atan2(x - cx, fx)
        position = self.get_position_from_scan_relative_to_base_scan(angle_offset)
        position = self.get_position_relative_to_map_from_base_scan(position)

        for index, previous_detection in enumerate(self.detections):
            if np.linalg.norm(np.array(previous_detection['position']) - np.array(position)) < 0.6:
                pred_id = f"{previous_detection['label']}_{index}"
                break
        else:
            self.get_logger().info(f"position: {position}")
            self.get_logger().info(f"label: {label}")
            pred_id = f"{label}_{len(self.detections)}"
            self.detections.append({
                "position": position,
                "label": label,
            })

        return pred_id

    def get_position_relative_to_map_from_base_scan(self, position):
        """
        Transforms a position from the base_scan frame to the map frame using TF2.
        """
        try:
            transform = self.tf_buffer.lookup_transform("map", "base_scan", rclpy.time.Time())
            point = PointStamped()
            point.header.frame_id = "base_scan"
            point.header.stamp = rclpy.time.Time().to_msg()
            point.point.x, point.point.y, point.point.z = position

            transformed_point = tf2_geometry_msgs.do_transform_point(point, transform)
            return (transformed_point.point.x, transformed_point.point.y, transformed_point.point.z)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f"Failed to transform position to map frame: {e}")
            return None

    def get_position_from_scan_relative_to_base_scan(self, angle_offset):
        """
        Uses the latest lidar scan to obtain a distance measurement for the given angle offset.
        Assumes that the lidar scan’s 0 angle corresponds to the forward direction.
        """
        scan = self.latest_scan

        self.get_logger().debug(f"Desired angle: {math.degrees(angle_offset)} degrees")
        self.get_logger().debug(f"Desired angle: {angle_offset} rad")
        self.get_logger().debug(f"Angle min: {scan.angle_min} rad")
        self.get_logger().debug(f"Angle increment: {scan.angle_increment} rad")
        # Find the index in the scan corresponding to the desired angle
        index = int(round((angle_offset - scan.angle_min) / scan.angle_increment))
        if index < 0 or index >= len(scan.ranges):
            self.get_logger().warn("Angle offset out of range for lidar scan")
            return None

        self.get_logger().debug(f"Index: {index}")
        self.get_logger().debug(f"Distance: {scan.ranges[index-5:index+6]}")
        subset = np.array(scan.ranges[index-5:index+6])
        filtered_subset = subset[np.isfinite(subset)]
        distance = np.mean(filtered_subset) if filtered_subset.size > 0 else float('nan')

        # Check for valid distance
        if distance < scan.range_min or distance > scan.range_max:
            self.get_logger().warn("Invalid lidar distance measurement")
            return None

        # Compute (x, y) in the lidar (robot) frame; assume z = 0.
        x = distance * math.cos(angle_offset)
        y = distance * math.sin(angle_offset)
        return (x, y, 0.0)

    def broadcast_tf(self, position, object_frame_id):
        """
        Broadcast a transform (TF) for the detected object.
        The parent frame is assumed to be "map".
        """
        t = geometry_msgs.msg.TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"
        t.child_frame_id = object_frame_id

        t.transform.translation.x = position[0]
        t.transform.translation.y = position[1]
        t.transform.translation.z = position[2]

        # For simplicity, we use an identity rotation.
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(t)
        self.get_logger().debug(f"Broadcasted TF for {object_frame_id} at {position}")

    def shutdown_callback(self):
        """
        Writes all detected objects (label and position) to a text file.
        """
        filename = "detections.txt"
        try:
            with open(filename, 'w') as file:
                for det in self.detections:
                    pos = det['position']
                    file.write(f"{det['label']}: x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}\n")
            self.get_logger().info(f"Detections saved to {filename}")
        except Exception as e:
            self.get_logger().error(f"Failed to write detections to file: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Call shutdown callback explicitly before shutting down.
        node.shutdown_callback()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
