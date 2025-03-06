#include "filter.h"
using namespace cv;
using namespace std;

//
//// EKF 3D estimation
//
tuple <Mat, Mat, vector<float>> EKF_3D(Mat Zvw, Mat xhat, double t, Mat Qk, Mat Rk, Mat Pk_1k_1)
{
	double px,py,pz, vx,vy,vz, q0,q1,q2,q3, wx,wy,wz;

	px = xhat.at<double>(0,0);
	py = xhat.at<double>(0,1);
	pz = xhat.at<double>(0,2);

	vx = xhat.at<double>(0,3);
	vy = xhat.at<double>(0,4);
	vz = xhat.at<double>(0,5);

	q0 = xhat.at<double>(0,6);
	q1 = xhat.at<double>(0,7);
	q2 = xhat.at<double>(0,8);
	q3 = xhat.at<double>(0,9);

	wx = xhat.at<double>(0,10);
	wy = xhat.at<double>(0,11);
	wz = xhat.at<double>(0,12);
	Mat p = (Mat_<double>(3,1) << px,py,pz);
	Mat v = (Mat_<double>(3,1) << vx,vy,vz);
	Mat q = (Mat_<double>(4,1) << q0,q1,q2,q3);
	Mat w = (Mat_<double>(3,1) << wx,wy,wz);
	double t2 = t*t/2;

	Mat Qrot = (Mat_<double>(3,3) << q0*q0+q1*q1-q2*q2-q3*q3,2*(q1*q2-q0*q3),2*(q0*q2+q1*q3),
																	2*(q0*q3+q1*q2),q0*q0-q1*q1+q2*q2-q3*q3,2*(-q0*q1+q2*q3),
																	2*(-q0*q2+q1*q3),2*(q2*q3+q0*q1),q0*q0-q1*q1-q2*q2+q3*q3);

	Mat Fk = (Mat_<double>(13,13) << 1,0,0, Qrot.at<double>(0,0)*t,Qrot.at<double>(1,0)*t,Qrot.at<double>(2,0)*t, 0,0,0,0, 0,0,0,
																	 0,1,0, Qrot.at<double>(0,1)*t,Qrot.at<double>(1,1)*t,Qrot.at<double>(1,2)*t, 0,0,0,0, 0,0,0,
																	 0,0,1, Qrot.at<double>(0,2)*t,Qrot.at<double>(1,2)*t,Qrot.at<double>(2,2)*t, 0,0,0,0, 0,0,0,

																	 0,0,0, 1,0,0, 0,0,0,0, 0,0,0,
																	 0,0,0, 0,1,0, 0,0,0,0, 0,0,0,
																	 0,0,0, 0,0,1, 0,0,0,0, 0,0,0,

																	 0,0,0, 0,0,0, 1,     -t/2*wx,-t/2*wy,-t/2*wz, 0,0,0,
																	 0,0,0, 0,0,0, t/2*wx, 1,     -t/2*wz, t/2*wy, 0,0,0,
																	 0,0,0, 0,0,0, t/2*wy, t/2*wz, 1,     -t/2*wx, 0,0,0,
																	 0,0,0, 0,0,0, t/2*wz,-t/2*wy, t/2*wx, 1,      0,0,0,

																	 0,0,0, 0,0,0, 0,0,0,0,                        1,0,0,
																	 0,0,0, 0,0,0, 0,0,0,0,                        0,1,0,
																	 0,0,0, 0,0,0, 0,0,0,0,                        0,0,1);

  Mat Gk = (Mat_<double>(13,6) <<  0,0,0, 0,0,0,
																	 0,0,0, 0,0,0,
																	 0,0,0, 0,0,0,

																	 t,0,0, 0,0,0,
																	 0,t,0, 0,0,0,
																 	 0,0,t, 0,0,0,

																	 0,0,0, 0,0,0,
																	 0,0,0, 0,0,0,
																	 0,0,0, 0,0,0,
																	 0,0,0, 0,0,0,

																	 0,0,0, t,0,0,
																 	 0,0,0, 0,t,0,
																 	 0,0,0, 0,0,t);

  Mat Hk = (Mat_<double>(6,13) <<  0,0,0, 1,0,0, 0,0,0,0, 0,0,0,
																	 0,0,0, 0,1,0, 0,0,0,0, 0,0,0,
																	 0,0,0, 0,0,1, 0,0,0,0, 0,0,0,
																	 0,0,0, 0,0,0, 0,0,0,0, 1,0,0,
																 	 0,0,0, 0,0,0, 0,0,0,0, 0,1,0,
																 	 0,0,0, 0,0,0, 0,0,0,0, 0,0,1);
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::normal_distribution<double> Qvelx(0.0,Qk.at<double>(0,0));
	std::normal_distribution<double> Qvely(0.0,Qk.at<double>(1,1));
	std::normal_distribution<double> Qvelz(0.0,Qk.at<double>(2,2));
	std::normal_distribution<double> Qomegax(0.0,Qk.at<double>(3,3));
	std::normal_distribution<double> Qomegay(0.0,Qk.at<double>(4,4));
	std::normal_distribution<double> Qomegaz(0.0,Qk.at<double>(5,5));
	Mat qvel = (Mat_<double>(3,1) <<  Qvelx(generator),Qvely(generator),Qvelz(generator));
	Mat qomega = (Mat_<double>(3,1) <<  Qomegax(generator),Qomegay(generator),Qomegaz(generator));

	Mat pkk_1, vkk_1, qkk_1, wkk_1, Id4, skeww, skewq;
	skeww = (Mat_<double>(4,4) <<  	1,     -t/2*wx,-t/2*wy,-t/2*wz,// -q1/2,-q2/2,-q3/2, //I4x4 + S(w)
																	t/2*wx, 1,     -t/2*wz, t/2*wy,//  q0/2, q3/2,-q2/2,
																	t/2*wy, t/2*wz, 1,     -t/2*wx,// -q3/2, q0/2, q1/2,
																	t/2*wz,-t/2*wy, t/2*wx, 1);     //  q2/2,-q1/2, q0/2);

	skewq = (Mat_<double>(4,3) <<  	-q1*t2,-q2*t2,-q3*t2,// -q1/2,-q2/2,-q3/2, //I4x4 + S(w)
																	 q0*t2, q3*t2,-q2*t2,//  q0/2, q3/2,-q2/2,
																	-q3*t2, q0*t2, q1*t2,// -q3/2, q0/2, q1/2,
																	 q2*t2,-q1*t2, q0*t2);     //  q2/2,-q1/2, q0/2);
	pkk_1 = p + Qrot*v*t;
	vkk_1 = v + qvel;
	qkk_1 = (skeww)*q;
	wkk_1 = w + qomega;
	//
	//Predict
	//
	Mat Xkk_1 = (Mat_<double>(13,1) << pkk_1.at<double>(0,0),
																		 pkk_1.at<double>(0,1),
																		 pkk_1.at<double>(0,2),

																		 vkk_1.at<double>(0,0),
																		 vkk_1.at<double>(0,1),
																		 vkk_1.at<double>(0,2),

																		 qkk_1.at<double>(0,0),
																		 qkk_1.at<double>(0,1),
																		 qkk_1.at<double>(0,2),
																		 qkk_1.at<double>(0,3),

																		 wkk_1.at<double>(0,0),
																		 wkk_1.at<double>(0,1),
																		 wkk_1.at<double>(0,2)); //Predicted state estimate

	Mat Pkk_1 = Fk*Pk_1k_1*Fk.t() + Gk*Qk*Gk.t(); //Predicted covariance estimate
	//
	//Update
	//
	std::normal_distribution<double> errx(0.0,Rk.at<double>(0,0));
	std::normal_distribution<double> erry(0.0,Rk.at<double>(1,1));
	std::normal_distribution<double> errz(0.0,Rk.at<double>(2,2));

	std::normal_distribution<double> errwx(0.0,Rk.at<double>(3,3));
	std::normal_distribution<double> errwy(0.0,Rk.at<double>(4,4));
	std::normal_distribution<double> errwz(0.0,Rk.at<double>(5,5));
	Mat hxkk = (Mat_<double>(6,1) << vkk_1.at<double>(0,0),vkk_1.at<double>(0,1),vkk_1.at<double>(0,2),wkk_1.at<double>(0,0),wkk_1.at<double>(0,1),wkk_1.at<double>(0,2));
	Mat zerr = (Mat_<double>(6,1) << errx(generator),erry(generator),errz(generator),errwx(generator),errwy(generator),errwz(generator));
	Mat zvw = Zvw + zerr;
	Mat eye13 = cv::Mat::eye(13,13,CV_64F);
	Mat yHilde = zvw - hxkk;
	Mat Sk = Hk*Pkk_1*Hk.t() + Rk; //Innovation (or residual) covariance
	Mat Kk = Pkk_1*Hk.t()*Sk.inv(); //Kalman gain
	Mat Xup = Xkk_1 + Kk*yHilde; //Updated state estimate
	Mat Pkk = (eye13 - Kk*Hk)*Pkk_1; //Updated covariance estimate
	vector<float> states;
	return {Xup.t(), Pkk, states};
}
