#include <iostream>
#include <vector>

//ROS
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

//Odometry
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Twist.h>

//CV
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/mat.hpp>
//Own-made functions
#include "function.h"
#include <fstream>

double iterations = 1;
Mat oldFrame, oldFrame0, oldFrame1, oldFrame2, frame, frame0, frame1, frame2, crop, oldCrop;

Mat R = cv::Mat::eye(3,3,CV_64F);
Mat R0 = cv::Mat::eye(3,3,CV_64F);
double tran[3][1] = {{0},{0},{0}};
double rot_help[3][1] = {{0},{0},{0}};
Mat Rpos = cv::Mat::eye(3,3,CV_64F);
Mat Rprev = cv::Mat::eye(3,3,CV_64F);
Mat tpos = Mat(3,1,CV_64F,tran);
Mat ttemp = tpos;
Mat rot = cv::Mat::eye(3,3,CV_64F);
Mat rotDiff = cv::Mat::eye(3,3,CV_64F);
Mat t = cv::Mat::zeros(cv::Size(1,3), CV_64F);
Mat tprev = cv::Mat::zeros(cv::Size(1,3), CV_64F);

Mat PkHat;
Mat Pk_1Hat = cv::Mat::eye(cv::Size(4,3), CV_64F);

vector<KeyPoint> keyp1, keyp2;
Mat desc1, desc2;
vector<DMatch> matches;
cv::Mat point3d;
vector<Point2d> scene1, scene2;

float yaw = 0;
float yawDiff = 0;
float yawVel;

int avgmatches = 0;
int nrmatches = 0;

double dim = 700;
double dimShow = 700;
double showScale = dimShow/dim;
double curScale = 1.4;
double prevScale = 1.4;
double alpha = 0.1;
double camHeight = 1.65;

cv::Rect crop_region(414, 175, 414, 200);

Mat trajectory = Mat::zeros(dim, dim, CV_8UC3);
Mat traj;
int X,Y; //Pixel location to print trajectory
double x,y,vx,vy;

Ptr<ORB> orbis = cv::ORB::create(1500,
		1.2f,
		8,
		15,
		0,
		3,
		ORB::FAST_SCORE,
	  31, // Descriptor patch size
		9);

double K_ar[3][3] = {
  {612.84,0,639.31},
  {0,612.8,367.35},
  {0,0,1}};
	double K_kitti[3][3] = {
	  {718.856,0,607.1928},
	  {0,718.856,185.2157},
	  {0,0,1}};
  Mat K = Mat(3,3,CV_64F,K_ar);
	Mat Kitti = Mat(3,3,CV_64F,K_kitti);
  double distCoeffs[8][1] = {0.3120601177215576,
                            -2.4365458488464355,
                             0.00019845466886181384,
                            -0.00034599119680933654,
                             1.4696191549301147,
                             0.19208934903144836,
                            -2.255640745162964,
                             1.3926784992218018};
  Mat dist = Mat(8,1,CV_64F,distCoeffs);

using namespace cv;
using namespace std;
static const std::string OPENCV_WINDOW = "Image Window";

class ImageConverter
{
public:
  ImageConverter()
    : it_(nh_)
  {
    // Subscribe to input video feed and publish output video feed
		image_sub_ = it_.subscribe("/kitti/camera_color_left/image_raw", 1, &ImageConverter::imageCb, this);

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
  //You can test your algorithm here.

	R.copyTo(Rprev);
	t.copyTo(tprev);

  if (iterations == 1)
  {
    oldFrame0 = cv_ptr->image;
    undistort(oldFrame0, oldFrame, K, dist, Mat());
		//undistort(oldFrame0, oldFrame, Kitti, Mat(), Mat());
		oldCrop = oldFrame(crop_region);
    orbis->detectAndCompute(oldCrop, noArray(), keyp1, desc1, false);
  }
    frame0 = cv_ptr->image;
    undistort(frame0, frame, K, dist, Mat());
		//undistort(frame0, frame, Kitti, Mat(), Mat());
		crop = frame(crop_region);
    orbis->detectAndCompute(crop, noArray(), keyp2, desc2, false);

		//cout << "Number of keypoints: " << keyp2.size() << endl;

    if(keyp1.size() > 6 || keyp2.size() > 6)
    {
    matches = BruteForce(oldCrop, crop, keyp1, keyp2, desc1, desc2, 0.5);
		//matches = BruteForce(oldCrop, crop, keyp1, keyp2, desc1, desc2, 0.5);
		//cout << "Matches size: " << matches.size() << endl;

    tie(t,R) = tranRot(keyp1, keyp2, matches);
		//hconcat(R,t,P);

		nrmatches = nrmatches + matches.size();
		avgmatches = nrmatches/iterations;
		double avgDist = avgMatchDist(matches);
		//cout << endl;

		PkHat = scaleUpdate(Kitti, Rprev, R, tprev, t);
		tie(scene1, scene2) = getPixLoc(keyp1, keyp2, matches);
		cv::triangulatePoints(Kitti*Pk_1Hat, PkHat, scene1, scene2, point3d);
		if(matches.size() < 500)
		{
		curScale = getScale(point3d, PkHat, matches, keyp2, prevScale, alpha, camHeight);
		prevScale = curScale;
	  }
		auto velocity = curScale/0.1;
		cout << "Total velocity: " << velocity << " m/s" << endl;
		if(R.rows == 3 && R.cols == 3 && t.rows == 3 && t.cols == 1 && avgDist > 10)
		{
		Rodrigues(Rpos, rot, noArray());
		Rodrigues(R, rotDiff, noArray());
		yawDiff = rotDiff.at<double>(1,0);
		yaw = rot.at<double>(1,0);
		//cout << "Rotation in degrees: " << endl << rot.at<double>(1,0)*180/3.14159 << endl;
    tpos = tpos + Rpos*t*curScale;
		Rpos = R*Rpos;
		}
		if(iterations > 1)
		{
    X = 2*tpos.at<double>(0,0);
    Y = 2*tpos.at<double>(0,2);

		x = -tpos.at<double>(0,2);
		y = tpos.at<double>(0,0);

		vx = -t.at<double>(0,2)*velocity;
		vy = t.at<double>(0,0)*velocity;

		yawVel = yawDiff/0.1;

		//cout << "Xpos: " << x << endl;
		//cout << "Ypos: " << y << endl;
		//cout << "xvec: " << vx << endl;
		//cout << "yvec: " << vy << endl;
		//cout << "Yaw: " << yaw*180/3.14159 << endl;
		//cout << "Yaw difference: " << yawDiff*180/3.14159 << endl;
	  }
		//since all odometry is 6DOF we'll need a quaternion created from yaw
    geometry_msgs::Quaternion odom_quat = tf::createQuaternionMsgFromYaw(yaw);

		//We'll publish the odometry message over ROS
    odom.header.frame_id = "odom";

    //set the position
    odom.pose.pose.position.x = x;
    odom.pose.pose.position.y = y;
    odom.pose.pose.position.z = 0.0;
    odom.pose.pose.orientation = odom_quat;
		odom.twist.twist.linear.x = vx;
		odom.twist.twist.linear.y = vy;
		odom.twist.twist.angular.z = yawVel;

		//cout << "Yaw: " << yaw << endl;
		//cout << "Average number of matches: " << avgmatches << endl;
		//cout << "Average match distance: " << avgDist << endl;

    //publish the message
    odom_pub_.publish(odom);

    //cout << "Iterations: " << iterations << endl;
		circle(trajectory, Point(X + dim/2,Y + dim/2), 2, Scalar(0,0,255), 2);
    cv::resize(trajectory, traj, cv::Size(), showScale, showScale);
    imshow( "Trajectory", traj );
    }

		oldCrop = crop.clone();
  	oldFrame = frame.clone();
    keyp1 = keyp2;
    desc1 = desc2.clone();

  iterations++;
//cout << "Frame number: " << iterations << endl;
    //------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------------------
    // Update GUI Window
    //cv::imshow(OPENCV_WINDOW, frame);
    cv::waitKey(1);
    // Output modified video stream
    //image_pub_.publish(cv_ptr->toImageMsg());

  }
private:
  ros::NodeHandle nh_;
	ros::NodeHandle n;
	ros::Publisher odom_pub_ = n.advertise<nav_msgs::Odometry>("odom", 1);
	nav_msgs::Odometry odom;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  //image_transport::Publisher image_pub_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_converter");
  ImageConverter ic;
  ros::spin();
  return 0;
}
