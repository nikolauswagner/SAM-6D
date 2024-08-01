#!/usr/bin/env python

import rospy
from std_msgs.msg import String

def talker():
    # Initialize the ROS node with the name 'talker'
    rospy.init_node('talker', anonymous=True)

    # Create a Publisher object, which will publish messages to the 'chatter' topic
    pub = rospy.Publisher('chatter', String, queue_size=10)

    # Set the rate at which messages are published (10 Hz)
    rate = rospy.Rate(10) # 10hz

    # Keep publishing messages until the node is shut down
    while not rospy.is_shutdown():
        # Create the message to be published
        hello_str = "hello world %s" % rospy.get_time()

        # Log the message to the console
        rospy.loginfo(hello_str)

        # Publish the message to the 'chatter' topic
        pub.publish(hello_str)

        # Sleep to maintain the loop rate
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
