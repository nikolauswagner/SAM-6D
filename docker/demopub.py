import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2

def read_and_publish_rgbd():
  """
  Reads an RGBD image from a file, separates color and depth channels,
  converts them to ROS Image messages, and publishes them on separate topics.
  """

  # Replace with your actual file paths
  color_image_path = "/workspace/docker/rgb.png"
  depth_image_path = "/workspace/docker/depth.png"

  rospy.init_node('rgbd_image_publisher', anonymous=True)
  bridge = CvBridge()

  # Read color and depth images
  try:
    color_image = cv2.imread(color_image_path)
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_GRAYSCALE)  # Preserve depth values
  except IOError as e:
    rospy.logerr(f"Error reading images: {e}")
    return

  # Convert images to ROS messages
  color_msg = bridge.cv2_to_imgmsg(color_image, encoding="bgr8")  # ROS-compatible color encoding
  depth_msg = bridge.cv2_to_imgmsg(depth_image, encoding="mono8")  # Preserve 16-bit depth
  color_info_msg = CameraInfo()
  color_info_msg.width = 640
  color_info_msg.height = 480
  color_info_msg.K = [525.5236157683187, 0.0, 317.3375061137099, 
                      0.0, 526.8173981134358, 252.9000179156094, 
                      0.0, 0.0, 1.0]
  color_info_msg.D = [0.0394987811046672, -0.09649876802553219, 0.005534059464925445, -0.0001591342485781845, 0.0]
  color_info_msg.R = [1.0, 0.0, 0.0, 
                      0.0, 1.0, 0.0, 
                      0.0, 0.0, 1.0]
  color_info_msg.P = [525.8593139648438, 0.0, 316.66318625247, 0.0, 
                      0.0, 527.5400390625, 254.2732262159698, 0.0, 
                      0.0, 0.0, 1.0, 0.0]
  color_info_msg.distortion_model = "plumbbob"  # Distortion model type

  # Create publishers for color and depth images (adjust topic names as needed)
  color_pub = rospy.Publisher("/xtion/rgb/image_raw", Image, queue_size=10)
  depth_pub = rospy.Publisher("/xtion/depth_registered/image", Image, queue_size=10)
  color_info_pub = rospy.Publisher("/xtion/rgb/camera_info", CameraInfo, queue_size=10)
  depth_info_pub = rospy.Publisher("/xtion/depth_registered/camera_info", CameraInfo, queue_size=10)

  # Publish images at a specified rate (adjust as needed)
  rate = rospy.Rate(10)  # 10 Hz

  while not rospy.is_shutdown():
    color_pub.publish(color_msg)
    depth_pub.publish(depth_msg)
    color_info_pub.publish(color_info_msg)
    depth_info_pub.publish(color_info_msg)
    rate.sleep()


if __name__ == '__main__':
  try:
    read_and_publish_rgbd()
  except rospy.ROSInterruptException:
    pass  # Handle ROS interruption gracefully
