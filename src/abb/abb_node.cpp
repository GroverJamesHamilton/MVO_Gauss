#include <iostream>
#include <vector>
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
//Own-made functions from function.cpp
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
int it = 1;
//Frame parameters for different purposes
Mat oldIm, oldIm0, Im, Im0, ImS, oldImS;
//Below are the rotation and translation matrices used for different purposes
Mat R = cv::Mat::eye(3,3,CV_64F); // Rotation matrix from epipolar geometry between 2 frames
Mat Rpos = cv::Mat::eye(3,3,CV_64F); // Rotation matrix
Mat Rprev = cv::Mat::eye(3,3,CV_64F);
Mat tpos = cv::Mat::zeros(cv::Size(1,3), CV_64F);
Mat rot = cv::Mat::eye(3,3,CV_64F);
Mat r = cv::Mat::eye(3,3,CV_64F);
Mat t = cv::Mat::zeros(cv::Size(1,3), CV_64F);
Mat tprev = cv::Mat::zeros(cv::Size(1,3), CV_64F);
Mat c2w = (Mat_<double>(3,3) << 0, 0, 1,
 															 -1, 0, 0,
																0,-1, 0);
//Features, descriptors, matches
vector<KeyPoint> keyp1, keyp2, keyp3, keyp4, keyp5, keyp6, testkeyp;
Mat desc1, desc2, desc3, desc4, desc5, desc6;
vector<DMatch> matchesOdom, matchesScale;
float pitchEKF, rollEKF, yawEKF;
//Triangulated 3D-points
cv::Mat point3d, pointX;
vector<Point2d> scene1, scene2, scene3, scene4, scene5, scene6;
//For publishing the yaw parameters
float yaw, yawDiff, yawVel, pitch, roll;
double dt = 0.1; //The sample time for the rosbag dataset [in s], default would be 0.1 or 0.03
//Display parameters
double dim = 500;
double dimShow = 1000;
double showScale = dimShow/dim;
//Scale recovery parameters
double s = 1;
double prevScale = 1;
double alpha = 1; //Scale smoothing parameters, alpha = [0 1], 1 is no smoothing
double acceptratio = 1;
double maxacc = 2;
double camHeight = 1.65; //Camera height, for scale recovery
vector<Point3d> Xground;
vector<Point2d> sift1, sift2;
//Scale recovery frame cropping
int xpix = 1242;
int ypix = 375;

int xOff = 500; //knop 500
int yOff = 225; //225
cv::Rect cropScale(xOff, yOff, xpix - 2*xOff, ypix - yOff); //Only a small subset of the frame is extracted
//The printed trajectory, Mainly for visually
Mat trajectory = Mat::zeros(dim, dim, CV_8UC3);
Mat traj;
int X,Y, Xekf, Yekf; //Pixel location to print trajectory
double x,y,z,vx,vy,vz, xekf, yekf, px, py;
double ErrScale = 0;
double errFilt = 0;
int iters = 0;
double timest = 0;
int GMMtraining = 0;
arma::gmm_diag model1;
arma::gmm_diag model2;
arma::gmm_diag model3;
//Initialize the ORB detectors/descriptors
//For the unscaled visual odometry
//Primarily: The max numbers of features and FAST threshold are the most crucial parameters
//to change since they have the most impact
Ptr<ORB> orbOdom = cv::ORB::create(750, //Max number of features
		1.25f,															 //Pyramid decimation ratio > 1
		8,																	 //Number of pyramid levels
		31,																	 //Edge threshold
		0,																	 //The level of pyramid to put source image to
		2,																	 //WTA_K
		ORB::FAST_SCORE,										 //Which algorithm is used to rank features, either FAST_SCORE or HARRIS_SCORE
	  31,                                  //Descriptor patch size
		11);                                  //The FAST threshold
//For the scale recovery knop
Ptr<ORB> orbScale = cv::ORB::create(1000, 1.1f, 12, 5, 0, 2, ORB::HARRIS_SCORE, 40, 25);

double fx = 721.5377; //903.7596 //721.5377
double fy = 721.5377; //901.9653 //721.5377
double ox = 609.5593; //695.7519 //609.5593
double oy = 172.8540; //224.2509 //172.8540

//The camera matrix dist. coeffs for the KITTI dataset
 double kOdom[3][3] = {
	  {fx,0,ox}, //Principal point needs to be offset if image is cropped
	  {0,fy,oy},
	  {0,0,1}};

 double kScale[3][3] = {
		{fx,0,ox - xOff}, //Principal point needs to be offset if image is cropped
		{0,fy,oy - yOff},
		{0,0,1}};

	Mat K = Mat(3,3,CV_64F,kOdom);
	Mat Kscale = Mat(3,3,CV_64F,kScale);
	Mat P1, P2;
	Mat normalVec;
	double n1,n2,n3,hprim, sig, mup;
	Mat canvas;
	Mat pnts3D;
	int err = 0;

// Ros time stamp
using namespace cv;
using namespace std;
static const std::string OPENCV_WINDOW = "Image Window";

double xGT, yGT, zGT, xGT2, yGT2, zGT2, scaleGT; //Ground truth parameters from node /tf
double yawGT, yawDiffGT;
double yawprevGT = -3.00251;
double velocity;
//Change parameters here:
bool scaleRecoveryMode = true;
int queueSize = 800; //800 knop
//Extended Kalman Filter ekf

Mat Xup; //State update
double errEkf = 0;
double errCalc = 0;
double mse = 0;
double errit = 0;
double pi = 3.14159265359;
//EKF Parameters
Mat xhat = (Mat_<double>(13,1) << 0,0,0, 4.5,-4.5,0, 0.9267984,0,0,-0.3755591, 0,0,0);
Mat Rk = (Mat_<double>(6,6) <<  0.1, 0, 0, 0,  0, 0,
															 	0, 0.1, 0, 0,  0, 0,
															 	0, 0, 0.01, 0,  0, 0,
															 	0, 0, 0, 0.00001, 0, 0,
																0, 0, 0, 0, 0.00001, 0,
															 	0, 0, 0, 0, 0, 0.00001);
Mat Qk = (Mat_<double>(6,6) <<  0.05, 0, 0, 0,  0, 0,
															 	0, 0.05, 0, 0,  0, 0,
															 	0, 0, 0.01, 0,  0, 0,
															 	0, 0, 0, 0.0005, 0, 0,
																0, 0, 0, 0, 0.0005, 0,
															 	0, 0, 0, 0, 0, 0.0005);

Mat Pkk = (Mat_<double>(13,13) << 0.05,0,0,0,0,0,0,0,0,0,0,0,0,
                                  0,0.05,0,0,0,0,0,0,0,0,0,0,0,
                                  0,0,0.06,0,0,0,0,0,0,0,0,0,0,

                                  0,0,0,0.025,0,0,0,0,0,0,0,0,0,
                                  0,0,0,0,0.025,0,0,0,0,0,0,0,0,
                                  0,0,0,0,0,0.04,0,0,0,0,0,0,0,

                                  0,0,0,0,0,0,0.03,0,0,0,0,0,0,
                                  0,0,0,0,0,0,0,0.03,0,0,0,0,0,
                                  0,0,0,0,0,0,0,0,0.03,0,0,0,0,
                                  0,0,0,0,0,0,0,0,0,0.03,0,0,0,

                                  0,0,0,0,0,0,0,0,0,0,0.01,0,0,
                                  0,0,0,0,0,0,0,0,0,0,0,0.01,0,
                                  0,0,0,0,0,0,0,0,0,0,0,0,0.02);

//Ground truth callback: Fetches values from node /tf
void tfCb(const tf2_msgs::TFMessage::ConstPtr& tf_msg)
{
//Ground truth translation
xGT = tf_msg->transforms.at(0).transform.translation.x;
yGT = tf_msg->transforms.at(0).transform.translation.y;
zGT = tf_msg->transforms.at(0).transform.translation.z;

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
//cout << xGT << "," << yGT << "," << zGT << "," << RollGT << "," << PitchGT << "," << YawGT << ";" << endl;
//cout << RollGT << "," << PitchGT << "," << YawGT << ";" << endl;
auto yawDiffGT = yawGT - yawprevGT;//cout << yawDiffGT << endl;//Ground truth scale
scaleGT = sqrt(pow((xGT - xGT2),2)+pow((yGT - yGT2),2)+pow((zGT - zGT2),2));

//Prints ground truth to window
circle(trajectory, Point(yGT + dim/2,xGT + dim/2), 1, Scalar(124,252,0), 1);
//Saves previous parameters of interest
yawprevGT = yawGT;
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
    // Subscribe to input video feed and publish output video feed
															//Replace this with your topic below//
		image_sub_ = it_.subscribe("/kitti/camera_gray_left/image_raw", queueSize, &ImageConverter::imageCb, this);
    //cv::namedWindow(OPENCV_WINDOW);
  }
  ~ImageConverter()
  {
    //cv::destroyWindow(OPENCV_WINDOW);
  }
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
	//------------------------------------------------------------------------------------------
  //Algorithm starts here
  std::chrono::steady_clock::time_point starttime = std::chrono::steady_clock::now();
  if(it == 1) //First frame to initialize
  {
    oldIm = cv_ptr->image; //Receive image
		orbOdom->detectAndCompute(oldIm, noArray(), keyp1, desc1, false);
		//Same with a subset of the frame if the scale recovery is wanted
		Mat Rinit;
		Rodrigues(Rpos, Rinit, noArray());
		Rinit.at<double>(1,0) = 0.7826064; //36
		Rodrigues(Rinit, Rpos, noArray());
		if(scaleRecoveryMode)
		{
		oldImS = oldIm(cropScale);
		//orbScale->detectAndCompute(oldImS, noArray(), keyp3, desc3, false);
		keyp3 = shiTomasiHelp(oldImS, 10000, 0.01, 5, 3, false, 0.04); //knop
		orbScale->compute(oldImS, keyp3, desc3);
	  }
  }

    Im = cv_ptr->image;
		orbOdom->detectAndCompute(Im, noArray(), keyp2, desc2, false);
		if(scaleRecoveryMode)
		{
		ImS = oldIm(cropScale);
		//orbScale->detectAndCompute(oldImS, noArray(), keyp4, desc4, false);
    keyp4 = shiTomasiHelp(ImS, 10000, 0.01, 5, 3, false, 0.04);
    orbScale->compute(ImS, keyp4, desc4);
	  }
      matchesOdom = BruteForce(oldIm, Im, keyp1, keyp2, desc1, desc2, 0.75, 1); //0.75
      matchesScale = BruteForce(oldImS, ImS, keyp3, keyp4, desc3, desc4, 0.9, 1); //0.99
		if(matchesOdom.size() < 10000)
    //if(keyp1.size() > 6 || keyp2.size() > 6) //Segmentation fault if detected features are lower than 5
    {
		//For triangulation, the matching 2D-correspondences from both frames are collected
		tie(scene1, scene2) = getPixLoc(keyp1, keyp2, matchesOdom);
		tie(scene3, scene4) = getPixLoc(keyp3, keyp4, matchesScale);
		//Initial epipolar geometry between the 2 current frames
    tie(t,R) = getInitPose(keyp1, keyp2, matchesOdom, K);
		Rodrigues(R, r, noArray()); //knop

      //if(it == 10)
			if(it > 2)
			{
				tie(P1, P2) = projMatrices(Kscale, R, t);
				cv::triangulatePoints(P1, P2, scene3, scene4, pnts3D);
				Xground = dim4to3(pnts3D);
        //inliers(Xground, 10);
        vector<size_t> inlierIndex;
        //knop
				tie(Xground, scene5, scene6) = siftError(P2, Xground, scene3, scene4, 3500); //2500
/*
      	bool stat1, stat2, stat3;
      	arma::mat data(1, Xground.size(), arma::fill::zeros);
      	for(int i = 0; i < Xground.size(); i++){data(0,i) = Xground[i].y;}
        if(!GMMtraining)
        {
          GMMtraining = 1;
          stat1 = model1.learn(data, 1, arma::eucl_dist, arma::random_subset, 10, 5, 1e-10, false);
          stat2 = model2.learn(data, 2, arma::eucl_dist, arma::random_subset, 10, 5, 1e-10, false);
          stat3 = model3.learn(data, 3, arma::eucl_dist, arma::random_subset, 10, 5, 1e-10, false);
        }
        else
        {
          std::chrono::steady_clock::time_point starttime = std::chrono::steady_clock::now();
          stat1 = model1.learn(data, 1, arma::eucl_dist, arma::keep_existing, 10, 5, 1e-10, false);
          stat2 = model2.learn(data, 2, arma::eucl_dist, arma::keep_existing, 10, 5, 1e-10, false);
          stat3 = model3.learn(data, 3, arma::eucl_dist, arma::keep_existing, 10, 5, 1e-10, false);
          std::chrono::steady_clock::time_point endtime = std::chrono::steady_clock::now();
          //cout << Xground.size() << "," << std::chrono::duration_cast<std::chrono::microseconds>(endtime - starttime).count() << ";" << endl;
        }
*/

				tie(Xground, sift1, sift2, inlierIndex) = siftPoints(Xground, P2, scene5, scene6, 300, false);

        //if(it > 2 && it % 10 == 0)
        if(it > 2)
        {
          Mat InlierIm = oldIm.clone();
          showInliersOutliers(inlierIndex, scene6, InlierIm, xOff, yOff, it);
        }
				if(sift1.size() > 5)
				{
        double acceptratio;
				vector <double> heights;
				tie(hprim, normalVec, heights, acceptratio, mup, sig) = generateHeights(Xground, 1.65, 10, 100);
        if(!isnan(hprim) && hprim > 0.1){s = camHeight/hprim;}
				}
        //cout << "Acceptratio: " << acceptratio << endl;
        //cout << s << ";" << endl;
        //cout << abs(s-scaleGT) << ";" << endl;
        //cout << endl;
        //cout << acceptratio << "," << abs(s-scaleGT) << "," << s << ";" << endl;
				//if(acceptratio > 0.6){s = camHeight/hprim;}
				//else{s = prevScale;}
        //cout << sig << "," << mup << "," << hprim << ";" << endl;
		  }
		Rodrigues(Rpos, rot, noArray()); 			//Converts rotation matrix to rotation vector
		Rodrigues(R, r, noArray());			      //Same as above but with the
		yawDiff = r.at<double>(1,0);
		yaw = rot.at<double>(1,0); 						//Yaw for publishing odom message
		roll = rot.at<double>(0,0);
		pitch = rot.at<double>(2,0);
		if(R.rows == 3 && R.cols == 3 && t.rows == 3 && t.cols == 1 && abs(yawDiff*180/3.14159) < 30)
		{
		if(scaleRecoveryMode && it > 2){tpos = tpos + Rpos*t*s;}//The scaled estimate is updated here
		else{tpos = tpos + Rpos*t;}															//The scaled estimate is updated here
		Rpos = R*Rpos;																					//The rotation matrix updated after
		}
}
		//EKF
		if(it > 2)
		{
    X = -tpos.at<double>(0,0);
    Y = -tpos.at<double>(0,2);
		x =  tpos.at<double>(0,2);
		y =  -tpos.at<double>(0,0);
		z =  -tpos.at<double>(0,1);
		Mat Pos = c2w*tpos;
		Mat V = c2w*t*s/dt;
		Mat Quat = rot2Quat(c2w*rot);
		Mat W = c2w*r/dt;
    Mat meas = (Mat_<double>(6,1) << V.at<double>(0,0),V.at<double>(0,1),V.at<double>(0,2), W.at<double>(0,0),W.at<double>(0,1),W.at<double>(0,2));
    //cout << x << "," << y << "," << z << "," << pitch << "," << roll << "," << yaw << ";" << endl;
		//cout << meas.t() << endl;
    cout << V.at<double>(0,0) << "," << V.at<double>(0,1) << "," << V.at<double>(0,2) << "," << W.at<double>(0,0) << "," << W.at<double>(0,1) << "," << W.at<double>(0,2) << ";" << endl;
	  vector<float> states;
		tie(xhat, Pkk, states) = EKF_3D(meas, xhat, dt, Qk, Rk, Pkk);
    px = xhat.at<double>(0,0);
    py = xhat.at<double>(0,1);
    vx = xhat.at<double>(0,3);
    vy = xhat.at<double>(0,4);
    vz = xhat.at<double>(0,5);
    //Mat Quater = (Mat_<double>(4,1) << xhat.at<float>(0,6),xhat.at<float>(0,7),xhat.at<float>(0,8),xhat.at<float>(0,9));
    if(it > 2)
    {
      //cout << xhat.at<double>(0,6) << " " << xhat.at<double>(0,7) << " " << xhat.at<double>(0,8) << " " << xhat.at<double>(0,9) << endl;
      tie(rollEKF,pitchEKF,yawEKF) = Quat2Euler( xhat.at<double>(0,6),  xhat.at<double>(0,7),  xhat.at<double>(0,8),  xhat.at<double>(0,9));
      //cout << yawEKF << endl;
      //cout << s << "," << sqrt(pow(vx,2) + pow(vy,2) + pow(vz,2))/10 << "," << scaleGT << ";" << endl;
      //double yawEKF = tf::getYaw(Quater);
      //cout << yawEKF << "," << yawGT << endl;
      //cout << Quater << endl;
      ErrScale = ErrScale + (s - scaleGT);
      //errFilt = errFilt + (sqrt(pow(vx,2) + pow(vy,2) + pow(vz,2))/10 - scaleGT);
      iters++;
      //cout << "Avg scale error: " << ErrScale/iters << endl;
      //cout << "Avg filt error: " << errFilt/iters << endl;
      //cout << "Accum scale error: " << ErrScale << endl;
      //cout << "Acc filt error: " << errFilt << endl;
      //cout << endl;
    }

    //cout << "x: " << x << " y: " << y << " z: " << z << endl;
    //cout << xGT << "," << yGT << "," << zGT << ";" << endl;
    //cout << "X: " << px << " Y: " << py << " Z: " << xhat.at<double>(0,2) << endl;
    //cout << "x: " << xGT << " y: " << yGT << " z: " << zGT << endl;
		//cout << endl;
	  }
		circle(trajectory, Point(X + dim/2,-Y + dim/2), 1, Scalar(0,0,255), 1);
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
		//Saving the previous epipolar geometry, mainly for the scale recovery
		Rprev = R.clone();
		tprev = t.clone();
		prevScale = s;
    it++; //Iteration number
    std::chrono::steady_clock::time_point endtime = std::chrono::steady_clock::now();
    //std::cout << "Calc time [ms]: " << std::chrono::duration_cast<std::chrono::milliseconds>(endtime - starttime).count() << std::endl;
    timest = timest + std::chrono::duration_cast<std::chrono::milliseconds>(endtime - starttime).count();
    //cout << "Avg est time: " << timest/it << endl;
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
	ros::NodeHandle nTf; //Ground truth nodehandle
	ros::Subscriber tf_msg = nTf.subscribe("/tf", queueSize, tfCb); //Ground truth message
  ImageConverter ic;
  ros::spin();
  return 0;
}

//Quaternion created from yaw to publish in nav_msgs
/*
geometry_msgs::Quaternion odom_quat = tf::createQuaternionMsgFromYaw(yaw);
//first, we'll publish the transform over tf
geometry_msgs::TransformStamped odom_trans;
//odom_trans.header.stamp = current_time;
odom_trans.header.frame_id = "odom";
odom_trans.child_frame_id = "base_link";
odom_trans.transform.translation.x = x;
odom_trans.transform.translation.y = y;
odom_trans.transform.translation.z = 0.0;
odom_trans.transform.rotation = odom_quat;

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
//Publish the odometry message
odom_pub_.publish(odom);
*/

//cout << "Normal vector: " << n1 << "," << n2 << "," << n3 << endl;
//cout << "Estimated height: " << hprim << endl;
/*
canvas = Mat::zeros(ImS.rows, ImS.cols, CV_8UC3);
for(int i = 0; i < scene5.size(); i++)
{
	std::string s = std::to_string(i);
	putText(canvas,s,Point(scene5[i].x,scene5[i].y), cv::FONT_HERSHEY_DUPLEX,0.3,Scalar(0,0,255),1);
	putText(canvas,s,Point(scene6[i].x,scene6[i].y), cv::FONT_HERSHEY_DUPLEX,0.3,Scalar(0,255,0),1);
}
cv::resize(canvas, canvas, cv::Size(), 3, 3);
imshow("Canvas", canvas);
*/
