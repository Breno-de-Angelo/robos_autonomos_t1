import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
import tf2_ros
import geometry_msgs.msg
import numpy as np
import open3d as o3d
import math
import atexit
from scipy.optimize import least_squares

def pointcloud2_to_xyz(msg):
    """
    Converts a sensor_msgs/PointCloud2 message to a (N x 3) numpy array.
    """
    points_list = []
    for point in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
        points_list.append([point[0], point[1], point[2]])
    return np.array(points_list, dtype=np.float32)

class ObjectDetector(Node):
    def __init__(self):
        super().__init__('object_detector')
        # Subscribe to the 3D lidar point cloud
        self.pc_sub = self.create_subscription(
            PointCloud2,
            '/velodyne2/velodyne_points2',  # Adjust this topic name as needed
            self.pc_callback,
            10)
        
        # Timer to process point clouds periodically
        self.detection_timer = self.create_timer(5.0, self.detect_items)
        self.tf_broadcasting_timer = self.create_timer(1.0, self.broadcast_all_tf)
    
        self.latest_pc_msg = None
        self.detections = []  # List to store detected objects

        # TF broadcaster for detected objects
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Ensure that shutdown_callback is called on exit
        atexit.register(self.shutdown_callback)

    def pc_callback(self, msg: PointCloud2):
        self.latest_pc_msg = msg

    def broadcast_all_tf(self):
        for index, det in enumerate(self.detections):
            self.get_logger().info("Broadcasting...")
            self.broadcast_tf(det['position'], f"{det['type']}_{str(index)}")

    def detect_items(self):
        self.broadcast_tf((0, 0, 0), "teste")
        if self.latest_pc_msg is None:
            self.get_logger().warn("No point cloud received yet.")
            return

        # Convert the ROS PointCloud2 message to a numpy array
        points = pointcloud2_to_xyz(self.latest_pc_msg)
        if points.size == 0:
            self.get_logger().warn("Empty point cloud.")
            return

        sphere_params, inliers = self.ransac_sphere(points, threshold=0.01, iterations=100)
        position = tuple(sphere_params[0:3])
        self.get_logger().info(f"size: {len(inliers)}")
        if len(inliers) > 200:
            frame_id = f"sphere_{len(self.detections)}"
            for index, det in enumerate(self.detections):
                if det['type'] == 'sphere':
                    if np.linalg.norm(np.array(det['position']) - np.array(position)) < 0.2:
                        break
            else:
                self.detections.append({'type': 'sphere', 'position': position, 'radius': sphere_params[3]})

    def fit_sphere(self, points):
        def residuals(params, points):
            x0, y0, z0, r = params
            return np.sqrt((points[:, 0] - x0)**2 + (points[:, 1] - y0)**2 + (points[:, 2] - z0)**2) - r

        # Initial guess for the sphere parameters
        x0, y0, z0 = np.mean(points, axis=0)
        r0 = np.mean(np.linalg.norm(points - [x0, y0, z0], axis=1))
        initial_guess = [x0, y0, z0, r0]

        result = least_squares(residuals, initial_guess, args=(points,))
        return result.x

    def ransac_sphere(self, points, threshold, iterations):
        best_inliers = []
        best_params = None

        for _ in range(iterations):
            sample = points[np.random.choice(points.shape[0], 4, replace=False)]
            params = self.fit_sphere(sample)
            inliers = []

            for point in points:
                distance = np.abs(np.sqrt((point[0] - params[0])**2 + (point[1] - params[1])**2 + (point[2] - params[2])**2) - params[3])
                if distance < threshold:
                    inliers.append(point)

            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_params = params

        return best_params, np.array(best_inliers)

    def broadcast_tf(self, position, object_frame_id):
        """
        Broadcasts a TF transform for the detected object.
        The parent frame is assumed to be "base_scan".
        """
        t = geometry_msgs.msg.TransformStamped()
        t.header.frame_id = "base_scan"
        t.child_frame_id = object_frame_id

        t.transform.translation.x = float(position[0])
        t.transform.translation.y = float(position[1])
        t.transform.translation.z = float(position[2])
        # Use an identity rotation (no orientation)
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        t.header.stamp = self.get_clock().now().to_msg()
        self.tf_broadcaster.sendTransform(t)
        self.get_logger().info(f"Broadcasted TF for {object_frame_id} at {position}")

    def convert_open3d_to_ros(self, pcd, frame_id="base_link"):
        """
        Converts an Open3D PointCloud to a sensor_msgs/PointCloud2 message.
        """
        points = np.asarray(pcd.points)
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = frame_id
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
        ]
        cloud_msg = pc2.create_cloud(header, fields, points)
        return cloud_msg

    def shutdown_callback(self):
        """
        Writes all detected objects (with type and parameters) to a text file on shutdown.
        """
        filename = "detections.txt"
        try:
            with open(filename, 'w') as f:
                for det in self.detections:
                    if det['type'] == 'sphere':
                        center = det['position']
                        radius = det['radius']
                        f.write(f"sphere: center=({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}), radius={radius:.2f}\n")
                    elif det['type'] == 'box':
                        center = det['position']
                        dims = det['dimensions']
                        f.write(f"box: center=({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}), dimensions=({dims[0]:.2f}, {dims[1]:.2f}, {dims[2]:.2f})\n")
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
        node.shutdown_callback()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
