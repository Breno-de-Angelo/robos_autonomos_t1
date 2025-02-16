import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import tf2_ros
import geometry_msgs.msg
import math
import atexit

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
            10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera_info',
            self.camera_info_callback,
            10)
        self.timer = self.create_timer(1.0, self.detect_items)
    
        self.bridge = CvBridge()
        self.latest_scan = None
        self.latest_image = None
        self.camera_info = None
        self.detections = []

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

    def detect_items(self):
        if self.camera_info is None:
            self.get_logger().warn("No camera info received yet. Skipping image processing.")
            return

        if self.latest_image is None:
            self.get_logger().warn("No image received yet. Skipping image processing.")
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error("CV Bridge error: {}".format(e))
            return

        debug_image = cv_image.copy() if self.debug else None
        image_width = self.camera_info.width
        fx = self.camera_info.k[0]
        cx = self.camera_info.k[2]

        # Convert to grayscale and blur for shape detection
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Detect Spheres (using Hough Circle Transform)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=100)

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                # Compute angle offset using camera intrinsics
                angle_offset = math.atan2(x - cx, fx)
                position = self.get_position_from_scan(angle_offset)
                if position:
                    frame_id = f"sphere_{len(self.detections)}"
                    for index, det in enumerate(self.detections):
                        if det['type'] == 'sphere':
                            if np.linalg.norm(np.array(det['position']) - np.array(position)) < 0.1:
                                self.broadcast_tf(position, f"{det['type']}_{str(index)}")
                                if self.debug:
                                    # Draw the detected circle and label it.
                                    cv2.circle(debug_image, (x, y), r, (0, 255, 0), 2)
                                    cv2.putText(debug_image, f"{det['type']}_{str(index)}", (x - 10, y - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                break
                    else:
                        self.broadcast_tf(position, frame_id)
                        self.detections.append({'type': 'sphere', 'position': position})
                        if self.debug:
                            # Draw the detected circle and label it.
                            cv2.circle(debug_image, (x, y), r, (0, 255, 0), 2)
                            cv2.putText(debug_image, frame_id, (x - 10, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Detect Boxes (by finding contours with 4 corners)
        edged = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                # Compute centroid of the contour
                M = cv2.moments(approx)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    angle_offset = math.atan2(cX - cx, fx)
                    position = self.get_position_from_scan(angle_offset)
                    if position:
                        frame_id = f"box_{len(self.detections)}"
                        for index, det in enumerate(self.detections):
                            if det['type'] == 'box':
                                if np.linalg.norm(np.array(det['position']) - np.array(position)) < 0.1:
                                    self.broadcast_tf(position, f"{det['type']}_{str(index)}")
                                    if self.debug:
                                        # Draw the box contour and label it.
                                        cv2.drawContours(debug_image, [approx], -1, (255, 0, 0), 2)
                                        # Compute approximate center for text placement.
                                        cY = int(M["m01"] / M["m00"])
                                        cv2.putText(debug_image, f"{det['type']}_{str(index)}", (cX - 10, cY - 10),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                                    break
                        else:
                            self.broadcast_tf(position, frame_id)
                            self.detections.append({'type': 'box', 'position': position})
                            if self.debug:
                                # Draw the box contour and label it.
                                cv2.drawContours(debug_image, [approx], -1, (255, 0, 0), 2)
                                # Compute approximate center for text placement.
                                cY = int(M["m01"] / M["m00"])
                                cv2.putText(debug_image, frame_id, (cX - 10, cY - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        if self.debug and debug_image is not None:
            try:
                debug_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding="bgr8")
                self.debug_pub.publish(debug_msg)
            except Exception as e:
                self.get_logger().error(f"Error publishing debug image: {e}")
        # exit()


    def get_position_from_scan(self, angle_offset):
        """
        Uses the latest lidar scan to obtain a distance measurement for the given angle offset.
        Assumes that the lidar scan’s 0 angle corresponds to the forward direction.
        """
        if self.latest_scan is None:
            self.get_logger().warn("No lidar scan received yet.")
            return None

        scan = self.latest_scan
        # Compute the desired angle (in radians) relative to the lidar scan's frame
        if angle_offset >= 0:
            desired_angle = angle_offset
        else:
            desired_angle = 2 * math.pi + angle_offset

        self.get_logger().info(f"Desired angle: {math.degrees(desired_angle)} degrees")
        # Find the index in the scan corresponding to the desired angle
        index = int(round((desired_angle - scan.angle_min) / scan.angle_increment))
        if index < 0 or index >= len(scan.ranges):
            self.get_logger().warn("Angle offset out of range for lidar scan")
            return None

        distance = scan.ranges[index]
        # Check for valid distance
        if distance < scan.range_min or distance > scan.range_max:
            self.get_logger().warn("Invalid lidar distance measurement")
            return None

        # Compute (x, y) in the lidar (robot) frame; assume z = 0.
        x = distance * math.cos(desired_angle)
        y = distance * math.sin(desired_angle)
        return (x, y, 0.0)

    def broadcast_tf(self, position, object_frame_id):
        """
        Broadcast a transform (TF) for the detected object.
        The parent frame is assumed to be "base_link".
        """
        t = geometry_msgs.msg.TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "base_link"
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
        self.get_logger().info(f"Broadcasted TF for {object_frame_id} at {position}")

    def shutdown_callback(self):
        """
        Writes all detected objects (type and position) to a text file.
        """
        filename = "detections.txt"
        try:
            with open(filename, 'w') as file:
                for det in self.detections:
                    pos = det['position']
                    file.write(f"{det['type']}: x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}\n")
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
