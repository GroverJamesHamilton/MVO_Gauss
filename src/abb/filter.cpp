#include "filter.h"
using namespace cv;
using namespace std;

tuple <Mat, Mat> EKF_filt(Mat Xmeas, Mat xhat, Mat Pk_1k_1, Mat Qk, Mat Rk)
{
	double x,y,vel,yaw,omega,xh,yh,velh,yawh,omegah;
	x = Xmeas.at<double>(0,0);
	y = Xmeas.at<double>(0,1);
	vel = Xmeas.at<double>(0,2);
	yaw = Xmeas.at<double>(0,3);
	omega = Xmeas.at<double>(0,4);
	xh = xhat.at<double>(0,0);
	yh = xhat.at<double>(0,1);
	velh = xhat.at<double>(0,2);
	yawh = xhat.at<double>(0,3);
	omegah = xhat.at<double>(0,4);
	double A, B, sA, sB, cA, cB;
	A = omegah/2;
	B = yawh + omegah/2;
	sA = sin(A);
	sB = sin(B);
	cA = cos(A);
	cB = cos(B);
	Mat Fk = (Mat_<double>(5,5) <<  1, 0, 2/omegah*sA*cB, -2*velh/omegah*sA*sB, (velh*omegah*cA*cB - velh*yawh*sA*sB - 2*velh*sA*cB)/pow(omegah,2),
																 	1, 0, 2/omegah*sA*sB, -2*velh/omegah*sA*cB, (velh*omegah*cA*sB - velh*yawh*sA*cB - 2*velh*sA*sB)/pow(omegah,2),
																 	0, 0, 1, 					  0, 													0,
																 	0, 0, 0,            1, 													1,
																 	0, 0, 0,            0, 													1);
  Mat Gk = (Mat_<double>(5,2) <<  2/omegah*sA*cB, (velh*omegah*cA*cB - velh*yawh*sA*sB - 2*velh*sA*cB)/pow(omegah,2),
																  2/omegah*sA*sB, (velh*omegah*cA*sB - velh*yawh*sA*cB - 2*velh*sA*sB)/pow(omegah,2),
																  1, 					 														 0,
																  0, 				   													   1,
																  0, 					 														 1);
  Mat Hk = (Mat_<double>(3,5) <<  1, 0, 0, 0, 0,
 																  0, 1, 0, 0, 0,
																  0, 0, 0, 1, 0);
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::normal_distribution<double> distvel(0.0,Qk.at<double>(0,0));
	std::normal_distribution<double> distyawvel(0.0,Qk.at<double>(1,1));
	double xkk_1,ykk_1,velkk_1,yawkk_1,omegakk_1;
	xkk_1 = xh + velh*cos(yawh);
	ykk_1 = yh + velh*sin(yawh);
	velkk_1 = velh + distvel(generator);
	yawkk_1 = yawh + omegah;
	omegakk_1 = omegah + distyawvel(generator);
	//
	//Predict
	//
	Mat Xkk_1 = (Mat_<double>(5,1) << xkk_1, ykk_1, velkk_1, yawkk_1, omegakk_1); //Predicted state estimate
	Mat Pkk_1 = Fk*Pk_1k_1*Fk.t() + Gk*Qk*Gk.t(); //Predicted covariance estimate
	//
	//Update
	//
	Mat eye5x5 = cv::Mat::eye(5,5,CV_64F);
	double zx, zy, zyaw;
	std::normal_distribution<double> errx(0.0,Rk.at<double>(0,0));
	std::normal_distribution<double> erry(0.0,Rk.at<double>(1,1));
	std::normal_distribution<double> erryaw(0.0,Rk.at<double>(2,2));
	zx = x + errx(generator);
	zy = y + erry(generator);
	zyaw = yaw + erryaw(generator);

	Mat yHilde = (Mat_<double>(3,1) <<  zx - xkk_1, zy - ykk_1, zyaw - yawkk_1);
	Mat Sk = Hk*Pkk_1*Hk.t() + Rk; //Innovation (or residual) covariance
	Mat Kk = Pkk_1*Hk.t()*Sk.inv(); //Kalman gain
	Mat Xup = Xkk_1 + Kk*yHilde; //Updated state estimate
	Mat Pkk = (eye5x5 - Kk*Hk)*Pkk_1; //Updated covariance estimate
	return {Xup.t(), Pkk};
}
//
//// EKF 3D estimation
//
tuple <Mat, Mat> EKF_3D(Mat meas, Mat xhat, double t, Mat Qk, Mat Rk, Mat Pk_1k_1)
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
	Mat Fk = (Mat_<double>(13,13) << 1,0,0, t,0,0, 0,0,0,0, 0,0,0,
																	 0,1,0, 0,t,0, 0,0,0,0, 0,0,0,
																	 0,0,1, 0,0,t, 0,0,0,0, 0,0,0,

																	 0,0,0, 1,0,0, 0,0,0,0, 0,0,0,
																	 0,0,0, 0,1,0, 0,0,0,0, 0,0,0,
																	 0,0,0, 0,0,1, 0,0,0,0, 0,0,0,

																	 0,0,0, 0,0,0, 1,     -t/2*wx,-t/2*wy,-t/2*wz, -q1/2,-q2/2,-q3/2,
																	 0,0,0, 0,0,0, t/2*wx, 1,     -t/2*wz, t/2*wy,  q0/2, q3/2,-q2/2,
																	 0,0,0, 0,0,0, t/2*wy, t/2*wz, 1,     -t/2*wx, -q3/2, q0/2, q1/2,
																	 0,0,0, 0,0,0, t/2*wz,-t/2*wy, t/2*wx, 1,       q2/2,-q1/2, q0/2,

																	 0,0,0, 0,0,0, 0,0,0,0,                         1,    0,    0,
																	 0,0,0, 0,0,0, 0,0,0,0,                         0,    1,    0,
																	 0,0,0, 0,0,0, 0,0,0,0,                         0,    0,    1);

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

  Mat Hk = (Mat_<double>(3,13) <<  1,0,0, 0,0,0,0, 0,0,0, 0,0,0,
																	 0,1,0, 0,0,0,0, 0,0,0, 0,0,0,
																	 0,0,1, 0,0,0,0, 0,0,0, 0,0,0);

	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::normal_distribution<double> Qvel(0.0,Qk.at<double>(0,0));
	std::normal_distribution<double> Qomega(0.0,Qk.at<double>(3,3));
	Mat qvel = (Mat_<double>(3,1) <<  Qvel(generator),Qvel(generator),Qvel(generator));
	Mat qomega = (Mat_<double>(3,1) <<  Qomega(generator),Qomega(generator),Qomega(generator));

	Mat pkk_1, vkk_1, qkk_1, wkk_1, Id4, skeww;
	skeww = (Mat_<double>(4,4) <<  	1,     -t/2*wx,-t/2*wy,-t/2*wz, -q1/2,-q2/2,-q3/2, //I4x4 + S(w)
																	t/2*wx, 1,     -t/2*wz, t/2*wy,  q0/2, q3/2,-q2/2,
																	t/2*wy, t/2*wz, 1,     -t/2*wx, -q3/2, q0/2, q1/2,
																	t/2*wz,-t/2*wy, t/2*wx, 1,       q2/2,-q1/2, q0/2);
	pkk_1 = p + v*t;
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
	Mat zp;
	std::normal_distribution<double> errx(0.0,Rk.at<double>(0,0));
	std::normal_distribution<double> erry(0.0,Rk.at<double>(1,1));
	std::normal_distribution<double> errz(0.0,Rk.at<double>(2,2));
	Mat zerr = (Mat_<double>(3,1) << errx(generator),erry(generator),errz(generator));
	zp = meas + zerr;
	Mat eye13 = cv::Mat::eye(13,13,CV_64F);
	Mat yHilde = zp - pkk_1;
	Mat Sk = Hk*Pkk_1*Hk.t() + Rk; //Innovation (or residual) covariance
	Mat Kk = Pkk_1*Hk.t()*Sk.inv(); //Kalman gain
	Mat Xup = Xkk_1 + Kk*yHilde; //Updated state estimate
	Mat Pkk = (eye13 - Kk*Hk)*Pkk_1; //Updated covariance estimate
	return {Xup.t(), Pkk};
}
