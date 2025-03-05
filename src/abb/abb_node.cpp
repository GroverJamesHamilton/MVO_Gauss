//Written by Alexander Smith, 2022-2023

#include <iostream>
#include <vector>
#include <unistd.h>
// ROS
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/CameraInfo.h>
// Odometry node handling
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Twist.h>
// OpenCV
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
//Own-made functions from function.cpp and filter.cpp (EKF)
#include "function.h"
#include "filter.h"
//For tf, ground truth data in KITTI datasets are stored here
#include <tf/transform_listener.h>
#include "tf/message_filter.h"
#include "message_filters/subscriber.h"
#include <tf2_ros/transform_listener.h>
#include <opencv2/viz.hpp>
#include <random>
//Algorithm iteration index
uint16_t it = 1;
//Frame parameters for different purposes
Mat oldIm, Im, ImS, oldImS;
//Below are the rotation and translation matrices used for different purposes
Mat Rmat = cv::Mat::eye(3,3,CV_64F);              //Rotation matrix from epipolar geometry between 2 frames
Mat Rpos = cv::Mat::eye(3,3,CV_64F);              //Current rotation
Mat tpos = cv::Mat::zeros(cv::Size(1,3), CV_64F); //Current position
Mat rotvec = cv::Mat::eye(3,3,CV_64F);            //Epipolar rotation vector
Mat rvec = cv::Mat::eye(3,3,CV_64F);              //
Mat tvec = cv::Mat::zeros(cv::Size(1,3), CV_64F); //
//Camera to world coordinate conversion (According to the REP-103 standard)
Mat c2w = (Mat_<double>(3,3) << 0, 0, 1,
 															 -1, 0, 0,
																0,-1, 0);
//Features, descriptors, matches
vector<KeyPoint> keyp1, keyp2, keyp3, keyp4;//1 and 2 for full image, 3 and 4 for ROI
Mat desc1, desc2, desc3, desc4;             //Same as above
vector<DMatch> matchesOdom, matchesScale;   //Stored feature matches
float pitchEKF, rollEKF, yawEKF;
//2D-coords of feature matches, scene5 and 6 sifted from 3 and 4 (Outlier rejected w.r.t. reprojection error)
vector<Point2d> scene1, scene2, scene3, scene4, scene5, scene6;
//For publishing the yaw parameters
float yaw, yawDiff, yawVel, pitch, roll;
//Display parameters
double dim = 500;         //Trajectory size [m]
double dimShow = 1000;    //Trajectory window size [px]
double showScale = dimShow/dim;
//Scale recovery parameters
double scaleParam = 1;    //Initial scale parameter
double acceptratio;
vector<Point3d> Xground; //Triangulated 3D-points
vector<Point2d> sift1,sift2; //Sifted points
//Scale recovery ROI (Region Of Interest) frame cropping
int xpix = 1242;    //Orig. Image width (px)
int ypix = 375;     //Orig. Image height (py)
//ROI offset
int xOff = 500;
int yOff = 225;//225;
cv::Rect cropScale(xOff, yOff, xpix - 2*xOff, ypix - yOff); //Only a small subset of the frame is extracted
//The printed trajectory, Mainly for visually
Mat trajectory = Mat::zeros(dim, dim, CV_8UC3);
Mat traj;
int X,Y, Xekf, Yekf; //Pixel location to print trajectories
double x,y,z,vx,vy,vz, xekf, yekf, px, py;
double xGT, yGT, zGT, xGT2, yGT2, zGT2, scaleGT, yawGT; //Ground truth parameters from node /tf
vector<double> heights; //False heights were: scaleParam = camHeight/height
//
//// Change parameters here!
//
//_______________________________________________________________________________________________
//Focal Length and Principal Point of datasets
//Can be found dataset's <topic_name>/CameraInfo or
//in calibration files when downloading dataset
//(Type: "rosbag info <bagname>.bag" in terminal in the same directory)
//Focal Length
double fx = 721.5377;
double fy = 721.5377;
//Principal Point
double ox = 609.5593;
double oy = 172.8540;

double dt = 0.1;         //Sample time for the rosbag dataset [in s]
double camHeight = 1.65; //Camera height, for scale recovery

//Scale Parameter Recovery
bool scaleRecoveryMode = true;

double meantime = 0;

//Rostopic name for frames
string messtype = "/kitti/camera_gray_left/image_raw";

//Frame queue size, as large as the frame number for offline calculation
//1 for online
int queueSize = 800;

//Initial yaw direction (to correct for comparing estimates with ground truth)
double yawInit = -0.77;

//EKF Parameter states            x,y,z,vx,vy,vz,q0,q1,q2,q3,wx,wy,wz
Mat xhat = (Mat_<double>(13,1) << 0,0,0, 4.5,-4.5,0, 0.9267984,0,0,-0.3755591, 0,0,0);

//ROI parameters
//Feature detector, Shi-Tomasi:
double maxFeatures = 10000;
double featureQuality = 0.01;
double featureDistance = 3; //In pixels

//Lowe's feature matching ratio test
double LowesEp = 0.85;//For epipolar geometry
double LowesScale = 0.9;  //For scale recovery

//Maximum reprojection error
int maxRepError = 3500;
//Maximum points sifted
int maxSiftPoints = 300;
//The number of orders the model trains per dataset
int modelOrder = 4;
//Max hprim estimation iterations
int maxhIter = 25;

double rmse = 0;
double errsum = 0;
int errit = 0;

//_______________________________________________________________________________________________
//
//Initialize the ORB detectors/descriptors
//For the unscaled visual odometry
//Primarily: The max numbers of features and FAST threshold are the most crucial parameters
//to change since they have the most impact
Ptr<ORB> orbOdom = cv::ORB::create(1500, //Max number of features
		1.25f,															 //Pyramid decimation ratio > 1
		8,																	 //Number of pyramid levels
		31,																	 //Edge threshold
		0,																	 //The level of pyramid to put source image to
		2,																	 //WTA_K
		ORB::HARRIS_SCORE,									 //Which algorithm is used to rank features,
                                         //either FAST_SCORE (faster but less accurate) or HARRIS_SCORE (slower but more accurate)
	  20,                                  //Descriptor patch size
		15);                                 //The FAST threshold
//For the scale recovery (Descriptor only)
Ptr<ORB> orbScale = cv::ORB::create(1000, 1.1f, 12, 5, 0, 2, ORB::HARRIS_SCORE, 40, 25);

//The camera matrix dist. coeffs for the KITTI dataset
 double kOdom[3][3] = {
	  {fx,0,ox}, //Principal point needs to be offset if image is cropped
	  {0,fy,oy},
	  {0,0,1}};
//The camera matrix for the ROI
 double kScale[3][3] = {
		{fx,0,ox - xOff}, //Principal point needs to be offset if image is cropped
		{0,fy,oy - yOff},
		{0,0,1}};

Mat K = Mat(3,3,CV_64F,kOdom);
Mat Kscale = Mat(3,3,CV_64F,kScale);
Mat P1, P2; //Projection matrices
Mat normalVec;
double n1,n2,n3,hprim,sig,mup;
Mat pnts3D;

using namespace cv;
using namespace std;
static const std::string OPENCV_WINDOW = "Image Window";

//
////Extended Kalman Filter ekf
//
vector<float> states;
Mat vVec, wVec;
//EKF noise covariance matrix
Mat Rk = (Mat_<double>(6,6) <<  0.001, 0, 0, 0,  0, 0,
															 	0, 0.001, 0, 0,  0, 0,
															 	0, 0, 0.0001, 0,  0, 0,
															 	0, 0, 0, 0.00001, 0, 0,
																0, 0, 0, 0, 0.00001, 0,
															 	0, 0, 0, 0, 0, 0.00001);
//EKF process covariance matrix
Mat Qk = (Mat_<double>(6,6) <<  0.005, 0, 0, 0,  0, 0,
															 	0, 0.005, 0, 0,  0, 0,
															 	0, 0, 0.01,  0,  0, 0,
															 	0, 0, 0, 0.0005, 0, 0,
																0, 0, 0, 0, 0.0005, 0,
															 	0, 0, 0, 0, 0, 0.0005);
//EKF state covariance matrix
Mat Pkk = (Mat_<double>(13,13) << 0.06,0,0,0,0,0,0,0,0,0,0,0,0,
                                  0,0.05,0,0,0,0,0,0,0,0,0,0,0,
                                  0,0,0.06,0,0,0,0,0,0,0,0,0,0,

                                  0,0,0,0.025,0,0,0,0,0,0,0,0,0,
                                  0,0,0,0,0.025,0,0,0,0,0,0,0,0,
                                  0,0,0,0,0,0.040,0,0,0,0,0,0,0,

                                  0,0,0,0,0,0,0.03,0,0,0,0,0,0,
                                  0,0,0,0,0,0,0,0.03,0,0,0,0,0,
                                  0,0,0,0,0,0,0,0,0.03,0,0,0,0,
                                  0,0,0,0,0,0,0,0,0,0.03,0,0,0,

                                  0,0,0,0,0,0,0,0,0,0,0.01,0,0,
                                  0,0,0,0,0,0,0,0,0,0,0,0.01,0,
                                  0,0,0,0,0,0,0,0,0,0,0,0,0.02);

bool first = true;
bool stat = true;
bool startIm = false;

double t1 = 0;

//Ground truth callback: Fetches values from node /tf
void tfCb(const tf2_msgs::TFMessage::ConstPtr& tf_msg)
{
if(stat)
{
  //cout << "Iteration: " << it << endl;
  stat = false;
}

//Ground truth translation
xGT = tf_msg->transforms.at(0).transform.translation.x;
yGT = tf_msg->transforms.at(0).transform.translation.y;
zGT = tf_msg->transforms.at(0).transform.translation.z;

//Ground truth rotation
float rotw = tf_msg->transforms.at(0).transform.rotation.w;
float rotx = tf_msg->transforms.at(0).transform.rotation.x;
float roty = tf_msg->transforms.at(0).transform.rotation.y;
float rotz = tf_msg->transforms.at(0).transform.rotation.z;
auto q = tf_msg->transforms.at(0).transform.rotation;

float RollGT, PitchGT, YawGT;
tie(RollGT, PitchGT, YawGT) = Quat2Euler(rotw, rotx, roty, rotz);
double yawGT = tf::getYaw(q);
//Ground truth scale parameter
scaleGT = sqrt(pow((xGT - xGT2),2) + pow((yGT - yGT2),2) + pow((zGT - zGT2),2));
//Prints ground truth to window
circle(trajectory, Point(yGT + dim/2,xGT + dim/2), 1, Scalar(124,252,0), 1);
//Saves previous parameters of interest for scale parameter estimation
xGT2 = xGT;
yGT2 = yGT;
zGT2 = zGT;
}
class ImageConverter
{
public:
  ImageConverter()
    : it_(nh_)
  {
    //Subscribe to input video feed and publish output video feed
		image_sub_ = it_.subscribe(messtype, queueSize, &ImageConverter::imageCb, this);
  }
  ~ImageConverter(){}
	//Image callback, where the main algorithm is
  void imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
    cv_bridge::CvImagePtr cv_ptr;
		cv_bridge::CvImagePtr img_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

  //Algorithm starts here-------------------------------------------------------
  if(it == 1)//First frame to initialize
  {
    oldIm = cv_ptr->image; //Receive image
    //Detects features and corresponding descriptors of first image/frame
		orbOdom->detectAndCompute(oldIm, noArray(), keyp1, desc1, false);
    //Initialize heading in world coordinates
    //Important when comparing estimated and ground truth trajectories (pitch and roll assumed neglible)
		Mat Rinit;
		Rodrigues(Rpos, Rinit, noArray());//Convert first rotation matrix to vector
		Rinit.at<double>(1,0) = yawInit;  //Initialize yaw heading in rotation vector
		Rodrigues(Rinit, Rpos, noArray());//Update heading in rotation matrix
		if(scaleRecoveryMode)
		{
      //Detect and descript features in ROI
      oldImS = oldIm(cropScale);
      keyp3 = shiTomasiHelp(oldImS, maxFeatures, featureQuality, featureDistance, 3, false, 0.04);
      orbScale->compute(oldImS, keyp3, desc3);
	  }
  }
  //cout << scaleParam << endl;
    //Retrieve new image and match features
    Im = cv_ptr->image;
		orbOdom->detectAndCompute(Im, noArray(), keyp2, desc2, false);
		if(scaleRecoveryMode)
		{
      std::chrono::steady_clock::time_point start1 = std::chrono::steady_clock::now();
      ImS = oldIm(cropScale);
      keyp4 = shiTomasiHelp(ImS, maxFeatures, featureQuality, featureDistance, 3, false, 0.04);
      orbScale->compute(ImS, keyp4, desc4);
      std::chrono::steady_clock::time_point end1 = std::chrono::steady_clock::now();
      t1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count();
	  }
    //Matches features here
    matchesOdom = BruteForce(oldIm, Im, keyp1, keyp2, desc1, desc2, LowesEp, 1);
    std::chrono::steady_clock::time_point start2 = std::chrono::steady_clock::now();
    matchesScale = BruteForce(oldImS, ImS, keyp3, keyp4, desc3, desc4, LowesScale, 1);
    std::chrono::steady_clock::time_point end2 = std::chrono::steady_clock::now();
    double t2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count();
		if(matchesOdom.size() < 10000)
    {
      //For triangulation, the matching 2D-correspondences from both frames are collected
      tie(scene1, scene2) = getPixLoc(keyp1, keyp2, matchesOdom);
      tie(scene3, scene4) = getPixLoc(keyp3, keyp4, matchesScale);
      //Initial epipolar geometry between the 2 current frames, without scale recovery
      tie(tvec,Rmat) = getEpip(scene1, scene2, matchesOdom, K);
      Rodrigues(Rmat, rvec, noArray());
      std::chrono::steady_clock::time_point start3 = std::chrono::steady_clock::now();
			if(it == 10)//When 2 frames are processed, scale recovery can be performed
			{
				tie(P1, P2) = projMatrices(Kscale, Rmat, tvec);       //Returns projective matrices for triangulation
				cv::triangulatePoints(P1, P2, scene3, scene4, pnts3D);//Returns 4D-vector [x*sc,y*sc,z*sc,sc](sc = 3D scale parameter)
        vector<size_t> inlierIndex;
        //Converts 4D-vector to 3D and sifts out points with too large reprojection errors
        //Complementary to Lowe's ratio test
				tie(Xground, scene5, scene6) = siftError(P2, pnts3D, scene3, scene4, maxRepError);
        //Sifts out ground points from rest of outliers
				tie(Xground, sift1, sift2, inlierIndex) = siftPoints(Xground, P2, scene5, scene6, maxSiftPoints, modelOrder, false);
        //Show the ROI with resulting inliers and outliers
        showInliersOutliers(inlierIndex, scene6, oldIm, xOff, yOff, it);

				if(sift1.size() > 5)
				{
          //Generate the most optimal virtual height hprim
          tie(hprim, normalVec, heights, acceptratio, mup, sig) = generateHeights(Xground, 1.65, 10, maxhIter);
          if(!isnan(hprim) && hprim > 0.1){scaleParam = camHeight/hprim;}
				}
/*
        if(errit < 750)
        {
        errsum = errsum + pow(scaleParam-scaleGT,2);
        errit++;
        rmse = sqrt(errsum/errit);
        cout << "RMSE: " << rmse << endl;
        }
*/
		  }
      std::chrono::steady_clock::time_point end3 = std::chrono::steady_clock::now();
      double t3 = std::chrono::duration_cast<std::chrono::microseconds>(end3 - start3).count();
      meantime = meantime + t1 + t2 + t3;
      //cout << "Mean: " << meantime/((it-1)*1000000) << endl;
		Rodrigues(Rpos, rotvec, noArray()); 			//Converts rotation matrix to rotation vector
		Rodrigues(Rmat, rvec, noArray());			    //Same as above but with the
		yawDiff = rvec.at<double>(1,0);
		yaw = rotvec.at<double>(1,0); 						//Yaw for publishing odom message
		roll = rotvec.at<double>(0,0);
		pitch = rotvec.at<double>(2,0);
		if(Rmat.rows == 3 && Rmat.cols == 3 && tvec.rows == 3 && tvec.cols == 1 && abs(yawDiff*180/3.14159) < 30)
		{
      if(scaleRecoveryMode && it > 2){tpos = tpos + Rpos*tvec*scaleParam;}//The scaled estimate is updated here
      else{tpos = tpos + Rpos*tvec;}															        //The scaled estimate is updated here
      Rpos = Rmat*Rpos;																					          //The rotation matrix updated after
		}
}
		//EKF
		if(it > 1)
		{
      X = tpos.at<double>(0,0);
      Y = tpos.at<double>(0,2);
      x = tpos.at<double>(0,2);
      y = tpos.at<double>(0,0);
      z = tpos.at<double>(0,1);
      //For measurement update model
      vVec = c2w*tvec*scaleParam/dt;
      wVec = c2w*rvec/dt;
      Mat meas = (Mat_<double>(6,1) << vVec.at<double>(0,0),vVec.at<double>(0,1),vVec.at<double>(0,2), wVec.at<double>(0,0),wVec.at<double>(0,1),wVec.at<double>(0,2));
      //Iterate EKF-states
      tie(xhat, Pkk, states) = EKF_3D(meas, xhat, dt, Qk, Rk, Pkk);
      //Staes estimation for printing to trajectory
      px = xhat.at<double>(0,0);
      py = xhat.at<double>(0,1);
      vx = xhat.at<double>(0,3);
      vy = xhat.at<double>(0,4);
      vz = xhat.at<double>(0,5);
      //cout << xhat.at<double>(0,0) << "," << xhat.at<double>(0,1) << "," << xhat.at<double>(0,2) << endl;
      //cout << yaw << endl;
      //cout << vVec.at<double>(0,0) << "," << vVec.at<double>(0,1) << "," << vVec.at<double>(0,2) << "," << wVec.at<double>(0,0) << "," << wVec.at<double>(0,1) << "," << wVec.at<double>(0,2) << endl;
	  }
    //Draw and update trajectories
		circle(trajectory, Point(-Y + dim/2,-X + dim/2), 1, Scalar(0,0,255), 1);
		circle(trajectory, Point(py + dim/2,px + dim/2), 1, Scalar(255,165,0), 1);
    cv::resize(trajectory, traj, cv::Size(), showScale, showScale);
    imshow("Trajectory", traj);
		//Copying previous parameters such as keypoints, descriptors etc
  	oldIm = Im.clone();
    keyp1 = keyp2;
    desc1 = desc2.clone();
		oldImS = ImS.clone();
    keyp3 = keyp4;
    desc3 = desc4.clone();
    it++; //Iteration number
    //------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------------------
    cv::waitKey(1);
  }
private:
	//Nodehandles
  ros::NodeHandle nh_;  //Image nodehandle
	ros::NodeHandle n;	  //Odometry nodehandle
	image_transport::Subscriber info_sub_;
	ros::Publisher odom_pub_ = n.advertise<nav_msgs::Odometry>("odom", 1);
	nav_msgs::Odometry odom;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_converter");
  ros::NodeHandle nStat;  //Ground truth static, Initial
  ros::NodeHandle nTf;    //Ground truth nodehandle
  ros::Subscriber tf_msg = nTf.subscribe("/tf", queueSize, tfCb);      //Ground truth message
  ImageConverter ic;
  ros::spin();
  return 0;
}
