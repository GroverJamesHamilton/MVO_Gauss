#include <iostream>
#include <vector>

//ROS
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

//CV
#include <opencv2/opencv.hpp>


using namespace cv;
using namespace std;
static const std::string OPENCV_WINDOW = "Video";

class ImageConverter
{
public:
  ImageConverter()
    : it_(nh_)
  {
    // Subscribe to input video feed and publish output video feed
    image_sub_ = it_.subscribe("/kitti/camera_color_left/image_raw", 1, &ImageConverter::imageCb, this);
    //image_sub_ = it_.subscribe("/myumi_005/rgb/image_raw", 1, &ImageConverter::imageCb, this);
    image_pub_ = it_.advertise("/image_converter/output_video", 1);


    cv::namedWindow(OPENCV_WINDOW);

  }

  ~ImageConverter()
  {
    cv::destroyWindow(OPENCV_WINDOW);
  }

  void imageCb(const sensor_msgs::ImageConstPtr& msg)
  {


    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }


	//-------------------------------------------------------------------------------------------



    cv::Mat frame =cv_ptr->image;

    //You can test your algorithm here.


    //------------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------------------------



    // Update GUI Window
    cv::resize(frame, frame, cv::Size(), 0.5, 0.5);
    cv::imshow(OPENCV_WINDOW, frame);
    cv::waitKey(1);

    // Output modified video stream
    image_pub_.publish(cv_ptr->toImageMsg());
  }

private:
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_pub_;

};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_converter");
  ImageConverter ic;
  ros::spin();
  return 0;
}
