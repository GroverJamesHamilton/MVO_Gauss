#ifndef FILTER_H    // To make sure you don't declare the function more than once by including the header multiple times.
#define FILTER_H

#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"
#include <opencv2/core/mat.hpp>
#include <chrono>  // for high_resolution_clock
#include <tuple>
#include <vector>
#include <cstdlib>
#include <unistd.h>
#include <ros/ros.h>
#include <opencv2/viz.hpp>
#include <random>
#include <algorithm>
#include <iterator>
#include <armadillo>

using namespace cv;
using namespace std;

tuple <Mat, Mat, vector<float>> EKF_3D(Mat Zvw, Mat xhat, double t, Mat Qk, Mat Rk, Mat Pk_1k_1);

#endif
