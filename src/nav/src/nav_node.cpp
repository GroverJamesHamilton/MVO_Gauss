#include "ros/ros.h"
#include "sensor_msgs/NavSatFix.h"


void chatterCallback(const sensor_msgs::NavSatFix::ConstPtr& msg)
{
  ROS_INFO("Gps: lat: [%f], long: [%f]", msg->latitude,msg->longitude);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "nav_node");
  ros::NodeHandle n;
  ros::Subscriber sub = n.subscribe("/kitti/oxts/gps/fix", 1000, chatterCallback);
  ros::spin();
  return 0;
}
