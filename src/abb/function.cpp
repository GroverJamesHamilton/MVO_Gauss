#include "function.h"
using namespace cv;
using namespace std;
//
//Brute Force Matcher + displays matches
//
vector<DMatch> BruteForce(Mat img1, Mat img2, vector<KeyPoint> keyp1, vector<KeyPoint> keyp2, Mat desc1, Mat desc2, double ratio, int hamm) {
	Ptr<BFMatcher> matcher = BFMatcher::create(NORM_HAMMING2, false);
	if(hamm == 1)
	{
		matcher = BFMatcher::create(NORM_HAMMING, false);
	}
	vector<vector<DMatch>> matches;
	vector<DMatch> good_matches;
  if (keyp1.size() > 3 && keyp2.size() > 3) { //Minimum amount of features needed from both frames
	//Extract feature matches
	matcher->knnMatch(desc1, desc2, matches, 2);
	//Removes bad matches using the Lowe's ratio test
const float ratio_thresh = ratio;
for (size_t i = 0; i < matches.size(); i++)
{
		if (matches[i][0].distance < ratio_thresh * matches[i][1].distance)
		{
				good_matches.push_back(matches[i][0]);
		}
}
//Draw matches
Mat img_matches,img_matches_good;
drawMatches(img1, keyp1, img2, keyp2, good_matches, img_matches_good, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
cv::resize(img_matches_good, img_matches_good, cv::Size(), 2, 2);
imshow("Good Matches", img_matches_good);
}
	return {good_matches};
}

vector<DMatch> Brute(Mat img1, Mat img2, vector<KeyPoint> keyp1, vector<KeyPoint> keyp2, Mat desc1, Mat desc2, double ratio) {
//Makes sure descriptors are the right data type
	if(desc1.type()!=CV_32F) {
	    desc1.convertTo(desc1, CV_32F);
			//cout << "Converting to CV_32F" << endl;
	}
	if(desc2.type()!=CV_32F) {
	    desc2.convertTo(desc2, CV_32F);
			//cout << "Converting to CV_32F" << endl;
	}

	Ptr<BFMatcher> matcher = BFMatcher::create(NORM_L2, false);
	vector<vector<DMatch>> matches;
	vector<DMatch> good_matches;
  if (keyp1.size() > 3 && keyp2.size() > 3) { //Minimum amount of features needed from both frames
	//Extract feature matches
	matcher->knnMatch(desc1, desc2, matches, 2);
	//Removes bad matches using the Lowe's ratio test
const float ratio_thresh = ratio;
for (size_t i = 0; i < matches.size(); i++)
{
		if (matches[i][0].distance < ratio_thresh * matches[i][1].distance)
		{
				good_matches.push_back(matches[i][0]);
		}
}
//Draw matches
Mat img_matches,img_matches_good;
drawMatches(img1, keyp1, img2, keyp2, good_matches, img_matches_good, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
cv::resize(img_matches_good, img_matches_good, cv::Size(), 1, 1);
imshow("Good Matches", img_matches_good);
}
	return {good_matches};
}

//Returns epipolar geometry between 2 frames based on the RANSAC scheme
tuple <Mat, Mat> getInitPose(vector<KeyPoint> keyp1, vector<KeyPoint> keyp2, vector<DMatch> matches, Mat cam){
	vector<Point2f> scene1, scene2; //Pixel locations of feature matches
	Mat Em, Rm, t; //Essential matrix, Rotation matrix and translation vector
if (matches.size() > 5) { //5-point algorithm won't work with less

for( size_t i = 0; i < matches.size(); i++)
{
		//Retrieve the Pixel locations from the good matches
		scene1.push_back( keyp1[ matches[i].queryIdx ].pt);
		scene2.push_back( keyp2[ matches[i].trainIdx ].pt);
}

//Estimating the epipolar geometry between the 2 frames with a RANSAC scheme
Em = findEssentialMat(scene2, scene1, cam, RANSAC, 0.99, 1);
recoverPose(Em, scene2, scene1, cam, Rm, t, noArray());
}
return {t,Rm};
}

//Not currently used in this subset
//Bucketing: Divides frame into a nxm grid, detects features in each subset
//I.e. evenly distributes the keypoints more evenly
vector<KeyPoint> Bucketing(Mat img, int gridX, int gridY, int max)
{
	//The ORB-detector
	Ptr<ORB> ORBucket = cv::ORB::create(
		  120, // Max 120 features per grid point
			1.2f,//Pyramid decimation ratio
			8,	 //Nr of pyramid levels
			31,
			0,
			2,
			ORB::FAST_SCORE,
			31,
			5); //Detector
Mat crop;
vector <KeyPoint> helpKeyp;
//Divides image into grid
int gridRows = img.cols/gridX;
int gridCols = img.rows/gridY;
vector <KeyPoint> keyp;
//Detects features for each grid subset
for(size_t i = 0; i < gridX; i++)
{
	for(size_t j = 0; j < gridY; j++)
	{
		cv::Rect crop_region((i)*gridRows, (j)*gridCols, gridRows, gridCols);
		crop = img(crop_region);
		cout << "At point: " << (i)*gridRows << " , " << (j)*gridCols << endl;
		ORBucket->detect(crop, helpKeyp, Mat());
		//cout << "Number of keypoints generated: " << helpKeyp.size() << endl;
		//cout << endl;
		for(size_t k = 1; k < helpKeyp.size(); k++)
		{
			helpKeyp[k].pt.x = helpKeyp[k].pt.x + (i)*gridRows;
			helpKeyp[k].pt.y = helpKeyp[k].pt.y + (j)*gridCols;
		}
		//keyp.push_back(helpKeyp);
		keyp.insert(keyp.end(), helpKeyp.begin(), helpKeyp.end());
	}
}
//drawKeypoints(img, keyp, img);
//cv::imshow("Bucketing output", img);
cout << "Number of keypoints generated: " << keyp.size() << endl;
return keyp;
}
//
//// Scale Update
//
Mat scaleUpdate(Mat K, Mat R_, Mat R, Mat t_, Mat t){
//Mat R0 = cv::Mat::eye(3,3,CV_64F);
Mat A = R*R_.inv();
Mat B = t - R*R_.inv()*t_;
Mat proj;
hconcat(A,B,proj);
Mat PkHat = K*proj;
return proj;
}

Mat projMat(Mat cam, Mat rota, Mat t)
{
	Mat proj;
	hconcat(rota,t,proj);
	return cam*proj;
}

tuple <vector<Point2d>, vector<Point2d>> getPixLoc(vector<KeyPoint> keyp1, vector<KeyPoint> keyp2, vector<DMatch> matches){
	vector<Point2d> scene1, scene2;
for( size_t i = 0; i < matches.size(); i++)
{
		//-- Retrieve the keypoints from the good matches
		scene1.push_back( keyp1[ matches[i].queryIdx ].pt);
		scene2.push_back( keyp2[ matches[i].trainIdx ].pt);
}
return{scene1, scene2};
}

double avg(vector<double> v)
{
        double return_value = 0.0;
        int n = v.size();
        for(int i = 0;i < n; i++)
        {
          return_value += v[i];
        }
        return return_value/n;
}

//Function for variance, help function to get_Scale
double variance(vector<double> v,double mean)
{
        double sum = 0.0;
        double temp = 0.0;
        double var = 0.0;
				auto length = v.size();
        for (int j = 0; j < length; j++)
        {
            temp = pow(v[j] - mean,2);
            sum += temp;
        }
        var = sum/(length);
				return var;
}

//The main algorithm for scale recovery
double getScale(Mat point3d,vector<Point2d> scene, Mat PkHat, double prevScale, double alpha, double height){
int size;
double outlierFactor = 2; //Factor to remove outliers from the median value with repr. error lower than maxError
vector<double> Yval, YvalFiltered;
double X,Y,Z,W, x,y, median, avgY, newScale;
	for(int i = 0; i<point3d.cols; i++)
	{
		//Obtain Triangulated 3D-points and their 2D correspondence from last frame
		W = point3d.at<double>(3,i);
		Y = point3d.at<double>(1,i)/W;
		cout << "Ok: " << 1.65/Y << endl;
		Yval.push_back(Y);
	}
size = Yval.size();
cout << "Size: " << size << endl;
sort(Yval.begin(), Yval.end());
for(int i = 0; i < size; i++)
{
//cout << 1.65/Yval[i] << endl;
}
if(size > 2)
{
if(size % 2 == 0) //Size of valid Y-points are even
{
	median = (Yval[size/2-1] + Yval[size/2])/2;
}
if(size % 2 == 1) //Size of valid Y-points are odd
{
	median = Yval[Yval.size()/2];
}
for(int i = 0; i < size; i++){
	//Remove last outliers
	if(Yval[i] < outlierFactor*median && Yval[i] > median/outlierFactor)
	{
		YvalFiltered.push_back(Yval[i]);
	}
}
avgY = avg(YvalFiltered);
}
else
{
	avgY = avg(Yval);
}
if(size == 0)
{
	avgY = height/prevScale; //Return previous scale if no valid points are found
}
//Scale smoothing, no smoothing if input alpha = 1
newScale = (1 - alpha)*prevScale + alpha*height/avgY;
return newScale;
}

double avgMatchDist(vector<DMatch> matches)
{
double avg;
double sum = 0;
	for (size_t i = 0; i < matches.size(); i++)
	{
		sum = sum + matches[i].distance;
  }
	avg = sum/matches.size();
	return avg;
}

double correctTimeDivide(double timeDiff, double sampleTime)
{
if(timeDiff < 1)
	{
		return round(10*timeDiff)*sampleTime;
	}
else
	{
		return sampleTime;
	}
}

//knop
	tuple<vector<size_t>, vector<double>> inlierLikelihood(vector<double> yVal, double maxLH)
	{
		double bic1, bic2, bic3;
		//double aic1, aic2, aic3;
		bool status, status2, status3;
		int order = 1;
		double pi = 3.14159;
		double minScore = 10000;
		double mu = avg(yVal);
		double var = variance(yVal, mu);
		double sigma = sqrt(var);
		int N = yVal.size();
 		arma::gmm_diag model;
		arma::mat data(1, yVal.size(), arma::fill::zeros);

		for(int i = 0; i < yVal.size(); i++)
		{
			data(0,i) = yVal[i];
			//cout << yVal[i] << ";" << endl;
		}
		//cout << "Size: " << yVal.size() << endl;
		if(yVal.size() > 4)
		{
			status = model.learn(data, 1, arma::maha_dist, arma::random_subset, 10, 5, 1e-10, false);
			if(status)
			{
				//aic1 = 2 - 2*model.sum_log_p(data);
				bic1 = log(N) - 2*model.sum_log_p(data);
				if(bic1 < minScore)
				{
					minScore = bic1;
				}
			}
		status2 = model.learn(data, 2, arma::maha_dist, arma::random_subset, 10, 5, 1e-10, false);
		if(status2)
		{
			double mean0 = model.means(0);
			double mean1 = model.means(1);
			vector<double> means;
			means.push_back(mean0);
			means.push_back(mean1);
			auto it = minmax_element(begin(means), end(means));
			int min_idx = std::distance(means.begin(), it.first);
			//aic2 = 4 - 2*model.sum_log_p(data);
			bic2 = 2*log(N) - 2*model.sum_log_p(data);
			if(bic2 < minScore)
			{
				minScore = bic2;
				order = 2;
				//cout << "Changing model to 2" << endl;
				//mu = model.means[min_idx];
				//sigma = sqrt(model.dcovs[min_idx]);
			}
		}
		status3 = model.learn(data, 3, arma::maha_dist, arma::random_subset, 10, 5, 1e-10, false);
		if(status3)
		{
			double mean0 = model.means(0);
			double mean1 = model.means(1);
			double mean2 = model.means(2);
			vector<double> means;
			means.push_back(mean0);
			means.push_back(mean1);
			means.push_back(mean2);
			auto it = minmax_element(begin(means), end(means));
			int min_idx = std::distance(means.begin(), it.first);
			sort(means.begin(), means.end());
			mean0 = means[0];
			mean1 = means[1];
			mean2 = means[2];
			//aic3 = 6 - 2*model.sum_log_p(data);
			bic3 = 3*log(N) - 2*model.sum_log_p(data);
			if(bic3 < minScore)
			{
			order = 3;
			//cout << "Changing model to 3" << endl;
			//mu = model.means[min_idx];
			//sigma = sqrt(model.dcovs[min_idx]);
		  }
			}
			cout << "Bic: " << bic1 << " " << bic2 << " " << bic3 << endl;
			//cout << "Aic: " << aic1 << " " << aic2 << " " << aic3 << endl;
			//cout << "Order: " << order << endl;
			//cout << endl;
			}

	double z1 = -0.67;
	double z3 =  -z1;
	double Q1 = sigma*z1 + mu;
	double Q3 = sigma*z3 + mu;
	//cout << "Mean: " << mu << " Q1: " << Q1 << " Q3: " << Q3 << endl;
	//cout << "Points size: " << N << endl;
	double lh;
	vector<size_t> index;
	vector<double> lhoods;
	for(int i = 0; i < yVal.size(); i++)
	{
		lh = 1/(sqrt(2*pi)*sigma)*exp(-pow((yVal[i] - mu)/sigma, 2)/2);
		lhoods.push_back(lh);
		if(yVal[i] > Q1 && yVal[i] < Q3)
		{
			index.push_back(i);
		}
	}

	stable_sort(index.begin(), index.end(), [&lhoods](size_t i1, size_t i2) {return lhoods[i1] > lhoods[i2];});

	vector<double> likelihoods;
	for(int i = 0; i < yVal.size(); i++)
	{
		lh = 1/(sqrt(2*pi)*sigma)*exp(-pow((yVal[i] - mu)/sigma, 2)/2);
		likelihoods.push_back(lh);
	}

	vector<size_t> idx(yVal.size());
	iota(idx.begin(), idx.end(), 0);
	stable_sort(idx.begin(), idx.end(), [&likelihoods](size_t i1, size_t i2) {return likelihoods[i1] > likelihoods[i2];});
	return {index, likelihoods};
}
// Sifts out abnormaly large triangulated 3D-points and their 2D-point correspondences
tuple <vector<Point3d>,vector<Point2d>,vector<Point2d>> siftPoints(vector<Point3d> X3D, Mat proj, vector<Point2d> scene1, vector<Point2d> scene2, int maxPoints, bool show)
{
vector<double> yVal;
vector<Point2d> sort1, sort2;
vector<Point3d> coordsSifted;
double Ys;
for(int i = 0; i < X3D.size(); i++)
{
		Ys = X3D[i].y;
		yVal.push_back(Ys);
		//imp
		//cout << X3D[i].x << ", " << X3D[i].y << ", " << X3D[i].z << ";" << endl;
}
vector<size_t> index;
vector<double> likelihoods;
tie(index,likelihoods) = inlierLikelihood(yVal, 0.8);
//
//// Viz: Displays 3D-points, for frame analysis
//
int dispViz = 0;
if(dispViz)
{
viz::Viz3d window("3D-coords");
window.showWidget("coordinate", viz::WCoordinateSystem());
window.showWidget("points", viz::WCloud(X3D, viz::Color::green()));
window.spin();
}

if(maxPoints > index.size())
{
	maxPoints = index.size();
}
for(int n = 0; n < maxPoints; n++)
{
	if(n == yVal.size())
	{
		break;
	}
	//imp
	sort1.push_back(scene1[index[n]]);
	sort2.push_back(scene2[index[n]]);
	coordsSifted.push_back(X3D[index[n]]);
	cout << X3D[index[n]].y << endl;
	{
	}
}

return{coordsSifted, sort1, sort2};
}

tuple<Mat, Mat>projMatrices(Mat cam, Mat Rota, Mat tran)
{
	Mat M = (Mat_<double>(3,4) << 1.0, 0.0, 0.0, 0.0,
																0.0, 1.0, 0.0, 0.0,
																0.0, 0.0, 1.0, 0.0);
	Mat proj1 = cam*M;
	cv::vconcat(Rota,tran.t(),M);
	Mat proj2 = cam*M.t();
	return{proj1, proj2};
}

tuple<vector<Point3d>,vector<Point2d>,vector<Point2d>> siftError(Mat proj, vector<Point3d> Xtriang, vector<Point2d> scene1, vector<Point2d> scene2, int maxError)
{
	vector<Point3d> siftX;
	vector<Point2d> sift1, sift2;
	Mat Xk = cv::Mat::ones(cv::Size(1,4), CV_64F); //
	Mat xk = cv::Mat::ones(cv::Size(1,3), CV_64F);
	Mat Er; // Reprojection error matrix xk - Xk*P
	double e; // Resulting reprojection error
	for (size_t i = 0; i < Xtriang.size(); i++)
	{
	Xk.at<double>(0,0) = Xtriang[i].x;
	Xk.at<double>(0,1) = Xtriang[i].y;
	Xk.at<double>(0,2) = Xtriang[i].z;
	xk.at<double>(0,0) = scene2[i].x;
	xk.at<double>(0,1) = scene2[i].y;

	Er = xk - proj*Xk;
	e = sqrt(Er.at<double>(0,0)*Er.at<double>(0,0) + Er.at<double>(0,1)*Er.at<double>(0,1) + Er.at<double>(0,2)*Er.at<double>(0,2));

	if (e < maxError)
	{
		sift1.push_back(scene1[i]);
		sift2.push_back(scene2[i]);
		siftX.push_back(Xtriang[i]);
	}
}
	//cout << "New size: " << sift1.size() << endl;
	return{siftX, sift1, sift2};
}

tuple<vector<Point3d>,vector<Point2d>,vector<Point2d>> siftSelect(Mat proj, vector<Point3d> Xtriang, vector<Point2d> scene1, vector<Point2d> scene2, int maxError)
{
	vector<Point3d> siftX;
	vector<Point2d> sift1, sift2;
	Mat Xk = cv::Mat::ones(cv::Size(1,4), CV_64F); //
	Mat xk = cv::Mat::ones(cv::Size(1,3), CV_64F);
	Mat Er; // Reprojection error matrix xk - Xk*P
	double e; // Resulting reprojection error
	for (size_t i = 0; i < Xtriang.size(); i++)
	{
	Xk.at<double>(0,0) = Xtriang[i].x;
	Xk.at<double>(0,1) = Xtriang[i].y;
	Xk.at<double>(0,2) = Xtriang[i].z;
	xk.at<double>(0,0) = scene2[i].x;
	xk.at<double>(0,1) = scene2[i].y;

	Er = xk - proj*Xk;
	e = sqrt(Er.at<double>(0,0)*Er.at<double>(0,0) + Er.at<double>(0,1)*Er.at<double>(0,1) + Er.at<double>(0,2)*Er.at<double>(0,2));

	if (e < maxError)
	{
		sift1.push_back(scene1[i]);
		sift2.push_back(scene2[i]);
		siftX.push_back(Xtriang[i]);
	}
}
	//cout << "New size: " << sift1.size() << endl;
	return{siftX, sift1, sift2};
}

void disp(Mat Xg)
{
	double X,Y,Z,W;

		for(int i = 0; i< Xg.cols; i++){
			//Obtain Triangulated 3D-points and their 2D correspondence from last frame
			W = Xg.at<double>(i,3);
			X = Xg.at<double>(i,0)/W;
			Y = Xg.at<double>(i,1)/W;
			Z = Xg.at<double>(i,2)/W;
			cout << "X: " << X << " Y: " << Y << " Z: " << Z << endl;
			//cout << X + Y + Z << endl;
}
}
int binom(int n, int k)
{
   if (k == 0 || k == n)
   return 1;
   return binom(n - 1, k - 1) + binom(n - 1, k);
}

vector <vector <int>> combList(int n, int k)
{
		int len = binom(n, k);
		Mat c = cv::Mat::zeros(cv::Size(3,len), CV_8U);
		vector <vector <int>> list;
    std::string bitmask(k, 1); // k leading 1's
    bitmask.resize(n, 0); // n-k trailing 0's
    do {
				vector <int> temp;
        for (int i = 0; i < n; ++i) // [0..n-1] integers
        {
            if (bitmask[i])
						{
							temp.push_back(i);
						}
        }
				list.push_back(temp);
    } while (std::prev_permutation(bitmask.begin(), bitmask.end()));
		return list;
}

void medianAvg(vector <double> heights)
{
	double average;
	double median;
	double h;
	double outlierFactor = 1.5;
	int size = heights.size();
	vector<double> newH;
	if(size > 2)
	{
	if(size % 2 == 0) //Size of valid Y-points are even
	{
		median = (heights[size/2-1] + heights[size/2])/2;
	}
	if(size % 2 == 1) //Size of valid Y-points are odd
	{
		median = heights[heights.size()/2];
	}
  }
	for(int i = 0; i < heights.size(); i++)
	{
		h = heights[i];
		if(h < median*outlierFactor && h > median/outlierFactor)
		{
			newH.push_back(h);
		}
	}
	average = avg(newH);
	//cout << "Median: " << median << endl;
	cout << "Median avg: " << 1.65/average << endl;
}

tuple <double, int> qScore(vector <double> heights)
{
	double max = 0;
	double diff;
	double score;
	double diffSum;
	double mu = 50;
	int index;
	for(int i = 0; i < heights.size(); i++)
	{
		diffSum = 0;
	for(int j = 0; j < heights.size(); j++)
	{
		if(i != j)
		{
			diff = heights[i] - heights[j];
			score = exp(-mu*pow(diff,2));
			diffSum = diffSum + score;
		}
	}
if(diffSum > max)
{
	max = diffSum;
	index = i;
	//cout << "Best score is: " << heights[index] << endl;
	//cout << "With score: " << diffSum << endl;
}
}
//cout << "Best estimate: " << 1.65/heights[index] << endl;
return {heights[index], index};
}

tuple <Mat, Mat> EKF(double x, double y, double vel, double yaw, double omega, double dt, Mat Pkk, Mat Qk, Mat Rk)
{
	double xk, xkk, yk, ykk, velk, velkk, yawk, yawkk, omegak, omegakk;
	Mat Fk = (Mat_<double>(5,5) <<  1, 0, dt*cos(yaw), 	-vel*dt*sin(yaw)*omega, 		0,
																 	1, 0, dt*sin(yaw), 	-vel*dt*sin(yaw)*omega, 		0,
																 	0, 0, 1, 					  0, 													0,
																 	0, 0, 0,            1, 													0,
																 	0, 0, 0,            1, 													1);
  Mat Gk = (Mat_<double>(5,2) <<  dt*cos(yaw), 0,
																  dt*sin(yaw), 0,
																  1, 					 0,
																  0, 				   0,
																  0, 					 1);
  Mat Hk = (Mat_<double>(3,5) <<  1, 0, 0, 0, 0,
 																  0, 1, 0, 0, 0,
																  0, 0, 0, 1, 0);
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::normal_distribution<double> distvel(0.0,Qk.at<double>(0,0));
	std::normal_distribution<double> distyawvel(0.0,Qk.at<double>(1,1));
	xk = x + vel*dt*cos(yaw);
	yk = y + vel*dt*sin(yaw);
	velk = vel + distvel(generator);
	yawk = yaw + dt*omega;
	omegak = omega + distyawvel(generator);
	//
	//Predict
	//
	Mat Xk = (Mat_<double>(5,1) << xk, yk, velk, yawk, omegak); //Predicted state estimate
	Pkk = Fk*Pkk*Fk.t() + Gk*Qk*Gk.t(); //Predicted covariance estimate
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

	Mat yHilde = (Mat_<double>(3,1) <<  zx - xk, zy - yk, zyaw - yawk);
	Mat Sk = Hk*Pkk*Hk.t() + Rk; //Innovation (or residual) covariance
	Mat Kk = Pkk*Hk.t()*Sk.inv(); //Kalman gain
	Mat Xup = Xk + Kk*yHilde; //Updated state estimate
	Pkk = (eye5x5 - Kk*Hk)*Pkk; //Updated covariance estimate
	return {Xup, Pkk};
}

tuple <double, Mat, vector<double>> generateHeights(vector<Point3d> X, double realH, double maxScale, int maxIter)
{
Mat normVec = (Mat_<double>(1,3) << 0,1,0);
vector<Point3d> normal;
Mat point, N, N_norm;
Mat n = cv::Mat::zeros(3,3,CV_64F);
Mat l = cv::Mat::ones(3,1,CV_64F);
double norm, height, n1, n2, n3;
vector<double> heights;
vector <vector <int>> bin;

bin = combList(X.size(), 3); //Combination list of sizeof(X) choose 3 numbers
std::vector<int> myvector;
for(int i = 0; i < bin.size(); ++i){myvector.push_back(i);}
std::random_shuffle(myvector.begin(),myvector.end());

if(maxIter > bin.size()){maxIter = bin.size();}

if(X.size() > 2)
{
int u = 0;
for(int i = 0; i < maxIter; i++)
{
for(int j = 0; j < 3; j++)
{
	n.at<double>(j,0) = X[bin[myvector[i]][j]].x;
	n.at<double>(j,1) = X[bin[myvector[i]][j]].y;
	n.at<double>(j,2) = X[bin[myvector[i]][j]].z;
}
N = n.inv()*l;
norm = sqrt(pow(N.at<double>(0,0), 2) + pow(N.at<double>(1,0), 2) + pow(N.at<double>(2,0), 2));
N_norm = N/norm;
height = 1/norm;
n1 = N_norm.at<double>(0,0);
n2 = N_norm.at<double>(1,0);
n3 = N_norm.at<double>(2,0);
//imp
if(n2 < -0.95)
{
normal.push_back({n1,n2,n3});
heights.push_back(height);
//cout << 1.65/height << ";" << endl;
u++;
}
}
}
if(heights.size() > 5)
{
double average = avg(heights);
int index;
double h;
tie(h,index) = qScore(heights);
n1 = normal[index].x;
n2 = normal[index].y;
n3 = normal[index].z;
Mat normVec = (Mat_<double>(1,3) << n1,n2,n3);
//cout << "Best normal estimate: " << normal[index] << endl;
//cout << "Size: " << heights.size() << endl;
//cout << heights[index] << endl;
//medianAvg(heights);
//cout << "Average: " << average << endl;
return {h, normVec, heights};
}
return {0.1, normVec, heights};
}

void testHomog(vector<double> hestimates, vector<Point2d> scene2, Mat cam, vector<Point3d> Xground)
{
	Mat Hom, Xprim, Err;
	double sad, hcur, e, hmin;
	double minerror = 100000000000;
	Mat Xk = cv::Mat::ones(cv::Size(1,3), CV_64F);
	Mat xk = cv::Mat::ones(cv::Size(1,3), CV_64F);
	for(int i = 0; i < hestimates.size(); i++)
	{
		hcur = hestimates[i];
		Hom = (Mat_<double>(3,3) << 1, 0, 0,
																0, 0, -hcur,
																0, 1, 0);
		sad = 0;
		e = 0;
		for(int k = 0; k < Xground.size(); k++)
		{
			Xk.at<double>(0,0) = Xground[k].x;
			Xk.at<double>(0,1) = Xground[k].z;
			xk.at<double>(0,0) = scene2[k].x;
			xk.at<double>(0,1) = scene2[k].y;
			Err = xk - cam*Hom*Xk;
			e = sqrt(pow(Err.at<double>(0,0),2) + pow(Err.at<double>(0,1),2) + pow(Err.at<double>(0,1),2));
			sad = sad + e;
		}
		if(e < minerror)
		{
			hmin = hcur;
		}
	}
	cout << "Best estimate: " << 1.65/hmin << endl;
}



void orbDraw(Mat img)
{
	Ptr<ORB> ORBdraw = cv::ORB::create(1000,1.2f,8,5,0,2,ORB::FAST_SCORE,31,10);
	vector <KeyPoint> keyp;
	ORBdraw->detect(img, keyp, Mat());
	drawKeypoints(img, keyp, img);
	cv::resize(img, img, cv::Size(), 1.5, 1.5);
	cv::imshow("Keypoints ORB", img);
}

vector<KeyPoint> shiTomasi(Mat img)
{
	//bork
	vector<KeyPoint> keypoints;
	std::vector<cv::Point2f> corners;
	Mat gray;
	cvtColor(img, gray, cv::COLOR_BGR2GRAY);
	int maxCorners = 1000;
	double qualityLevel = 0.05;
  double minDistance = 3;
  cv::Mat mask;
	int blockSize = 3;
	bool useHarrisDetector = false;
  double k = 0.04;
	cv::goodFeaturesToTrack(gray, corners, maxCorners, qualityLevel, minDistance, mask, blockSize, useHarrisDetector, k);
	for(size_t i = 0; i < corners.size(); i++)
	{
	//cv::circle(img, corners[i], 3, cv::Scalar(255.), -1);
	}
	//cv::resize(img, img, cv::Size(), 1.5, 1.5);
  //cv::imshow("Keypoints Shi-Tomasi", img);
	for(size_t i = 0; i < corners.size(); i++)
	{
   keypoints.push_back(cv::KeyPoint(corners[i], 1.f));
	}
	return keypoints;
}

vector<KeyPoint> shiTomasiHelp(Mat img,
															 int maxCorners,
															 double qualityLevel,
															 int minDistance,
															 int blockSize,
															 bool useHarrisDetector,
														   double k)
{
	vector<KeyPoint> keypoints;
	std::vector<cv::Point2f> corners;
	Mat gray;
	cvtColor(img, gray, cv::COLOR_BGR2GRAY);
  cv::Mat mask;
	cv::goodFeaturesToTrack(gray, corners, maxCorners, qualityLevel, minDistance, mask, blockSize, useHarrisDetector, k);
	for(size_t i = 0; i < corners.size(); i++)
	{
	//cv::circle(img, corners[i], 3, cv::Scalar(255.), -1);
	}
	//cv::resize(img, img, cv::Size(), 1.5, 1.5);
  //cv::imshow("Keypoints Shi-Tomasi", img);
	for(size_t i = 0; i < corners.size(); i++)
	{
   keypoints.push_back(cv::KeyPoint(corners[i], 1.f));
	}
	return keypoints;
}

void PermGenerator(int n, int k)
{
    std::vector<int> d(n);
    std::iota(d.begin(),d.end(),1);
    cout << "These are the Possible Permutations: " << endl;
    do
    {
        for (int i = 0; i < k; i++)
        {
            cout << d[i] << " ";
        }
        cout << endl;
        std::reverse(d.begin()+k,d.end());
    } while (next_permutation(d.begin(),d.end()));
}

void nelderMead(Point3d normalVec, double h, Mat cam, vector<Point2d> scene1,vector<Point2d> scene2, Mat t, Mat rota)
{
	double n,n1,n2,n3,norm;
	Mat H;
	double nDiff = 0.2;
	double hDiff = 0.4;
	int iter = 100;
	double p = 2.5;
	double SAD;
	double min = 100000;
	double hbest;
	Mat x1 = cv::Mat::ones(cv::Size(1,3), CV_64F);
	Mat x2 = cv::Mat::ones(cv::Size(1,3), CV_64F);
	Mat err = cv::Mat::ones(cv::Size(1,3), CV_64F);
	Mat N = cv::Mat::zeros(cv::Size(3,1), CV_64F);
	n1 = normalVec.x;
	n2 = normalVec.y;
	n3 = normalVec.z;
	N.at<double>(0,0) = n1;
	N.at<double>(0,1) = n2;
	N.at<double>(0,2) = n3;
	h = h - hDiff/2;
	for(int i = 0; i < iter; i++)
	{
		h = h + hDiff/iter;
		n = n3;
		n = n - nDiff/2;
		for(int j = 0; j < iter; j++)
		{
			n = n + nDiff/iter;
			N.at<double>(0,2) = n;
			norm = sqrt(pow(n1,2)+pow(n2,2)+pow(n,2));
			N = N/norm;
			H = cam*(rota + t*N/h)*cam.inv();
			SAD = 0;
			for(int k = 0; k < scene1.size(); k++)
			{
				x1.at<double>(0,0) = scene1[k].x;
				x1.at<double>(1,0) = scene1[k].y;
				x2.at<double>(0,0) = scene2[k].x;
				x2.at<double>(1,0) = scene2[k].y;
				err = x2 - H*x1;
				auto e = sqrt(pow(err.at<double>(0,0),2) + pow(err.at<double>(0,1),2) + pow(err.at<double>(0,2),2));
				SAD = SAD + e;
			}
			if(SAD < min)
			{
				hbest = h;
				min = SAD;
				cout << "New best estimate: " << hbest << " i: " << i << " j: " << j << " SAD: " << SAD << endl;
			}
		}
	}
	cout << "New best estimate: " << 1.65/hbest << endl;
}

void nelder(Mat normalVec, double h, Mat cam, vector<Point2d> scene1,vector<Point2d> scene2, Mat t, Mat rota)
{
	double n,n1,n2,n3,norm, theta, phi, htemp, m1, m2, m3;
	Mat H;
	double pi = 3.14159265359;
	double phDiff = 10;
	double thDiff = 10;
	double hDiff = 2;
	int iterh = 10;
	int iterth = 50;
	int iterphi = 50;
	double p = 25;
	double SAD;
	double min = 100000;
	double hbest;
	Mat x1 = cv::Mat::ones(cv::Size(1,3), CV_64F);
	Mat x2 = cv::Mat::ones(cv::Size(1,3), CV_64F);
	Mat err = cv::Mat::ones(cv::Size(1,3), CV_64F);
	Mat N = cv::Mat::zeros(cv::Size(3,1), CV_64F);
	n1 = normalVec.at<double>(0,0);
	n2 = normalVec.at<double>(1,0);
	n3 = normalVec.at<double>(2,0);
	theta = acos(n3);
	phi = atan(n2/n1);

	theta = acos(n3) - thDiff*pi/360;
	for(int i = 0; i < iterth; i++)
	{
		theta = theta + thDiff*pi/360/iterth;
		phi = atan(n2/n1) - phDiff*pi/360;
		for(int j = 0; j < iterphi; j++)
		{
			phi = phi + phDiff*pi/360/iterphi;

			htemp = h - hDiff/2;
			for(int k = 0; k < iterh; k++)
			{
				htemp = htemp + hDiff/iterh;
				m1 = sin(theta)*cos(phi);
				m2 = sin(theta)*sin(phi);
				m3 = cos(theta);
				N.at<double>(0,0) = m1;
				N.at<double>(0,1) = m2;
				N.at<double>(0,2) = m3;
				H = cam*(rota + t*N/htemp)*cam.inv();
				SAD = 0;
				for(int v = 0; v < scene1.size(); v++)
				{
					x1.at<double>(0,0) = scene1[v].x;
					x1.at<double>(1,0) = scene1[v].y;
					x2.at<double>(0,0) = scene2[v].x;
					x2.at<double>(1,0) = scene2[v].y;
					err = x2 - H*x1;
					auto e = sqrt(pow(err.at<double>(0,0),2) + pow(err.at<double>(0,1),2) + pow(err.at<double>(0,1),2));
					SAD = SAD + e;
				}
				if(SAD < min)
				{
					hbest = htemp;
					min = SAD;
					//cout << k << endl;
					//cout << "New best estimate: " << hbest << " SAD: " << SAD << endl;
					cout << "New best estimate: " << hbest << " i: " << i << " j: " << j << " SAD: " << SAD << endl;
				}
			}
		}
	}
}


void spherical(Point3d normvec)
{
double nx, ny, nz, theta, phi, phi2;
cout << normvec << endl;
nx = normvec.x;
ny = normvec.y;
nz = normvec.z;
double pi = 3.14159265359;
theta = acos(nz);
phi = atan(ny/nx);

cout << sin(theta)*cos(phi) << " " << sin(theta)*sin(phi) << " " << cos(theta) << endl;
}


Mat rot2Quat(Mat rotvec)
{
	double pitch,roll,yaw,q0,q1,q2,q3;
	pitch = rotvec.at<double>(0,0);
	roll = rotvec.at<double>(0,1);
	yaw = rotvec.at<double>(0,2);

	q0 = cos(pitch/2)*cos(roll/2)*cos(yaw/2) + sin(pitch/2)*sin(roll/2)*sin(yaw/2);
	q1 = sin(pitch/2)*cos(roll/2)*cos(yaw/2) - cos(pitch/2)*sin(roll/2)*sin(yaw/2);
	q2 = cos(pitch/2)*sin(roll/2)*cos(yaw/2) + sin(pitch/2)*cos(roll/2)*sin(yaw/2);
	q3 = cos(pitch/2)*cos(roll/2)*sin(yaw/2) - sin(pitch/2)*sin(roll/2)*cos(yaw/2);

	Mat q = (Mat_<double>(4,1) <<  q0,q1,q2,q3);
	return q;
}



void dispProjError(Mat Proj, vector<Point2d> pt2, vector<Point3d> p3d)
{
	Mat errorMat;
	double error;
	vector<double> errors;
	double averageError;
	Mat Xk = cv::Mat::ones(cv::Size(1,4), CV_64F);
	Mat xk = cv::Mat::ones(cv::Size(1,3), CV_64F);
	for(int i = 0; i < p3d.size(); i++)
	{
		Xk.at<double>(0,0) = p3d[i].x;
		Xk.at<double>(0,1) = p3d[i].y;
		Xk.at<double>(0,2) = p3d[i].z;
		// Form the current Image-point from input with index i
		xk.at<double>(0,0) = pt2[i].x;
		xk.at<double>(0,1) = pt2[i].y;

		errorMat = xk - Proj*Xk;
		error = sqrt(errorMat.at<double>(0,0)*errorMat.at<double>(0,0) + errorMat.at<double>(0,1)*errorMat.at<double>(0,1) + errorMat.at<double>(0,2)*errorMat.at<double>(0,2));
		if(i < 10000)
		{
		//cout << "Reprojection error: " << error << endl;
	  }
		errors.push_back(error);
	}
	averageError = avg(errors);
	cout << "Average error: " << averageError << endl;
}

vector<Point3d> dim4to3(Mat dim4)
{
	//imp
vector<Point3d> p3d;
for(int i = 0; i < dim4.cols; i++)
{
		double xs = dim4.at<double>(0,i)/dim4.at<double>(3,i);
		double ys = dim4.at<double>(1,i)/dim4.at<double>(3,i);
		double zs = dim4.at<double>(2,i)/dim4.at<double>(3,i);
		p3d.push_back({xs,ys,zs});
}
return p3d;
}

Mat normalizeVec(Mat tvec)
{
	double normal;
	normal = pow(tvec.at<double>(0,0),2) + pow(tvec.at<double>(1,0),2) + pow(tvec.at<double>(2,0),2);
	return tvec/sqrt(normal);
}

tuple <Mat, Mat> refinePose(Mat cam, vector<Point2d> pt1, vector<Point2d> pt2, Mat Rt, Mat tt)
{
	vector<Point3d> p3d;
	Mat p4d(4, pt1.size(),CV_64F);
	Mat proj = (Mat_<double>(3,4) << 1.0, 0.0, 0.0, 0.0,
																   0.0, 1.0, 0.0, 0.0,
																   0.0, 0.0, 1.0, 0.0);
	Mat Proj1 = cam*proj;
	cv::vconcat(Rt,tt.t(),proj);
	Mat Proj2 = cam*proj.t();
	cv::triangulatePoints(Proj1, Proj2, pt1, pt2, p4d);
	vector<Point3d> Xd;
	vector<Point2d> s1;
	vector<Point2d> s2;
	//tie(Xd, s1, s2) = siftPoints(p4d, Proj2, pt1, pt2, 50, 10000, true);
	//bork
	p3d = dim4to3(p4d);
	tie(Xd, s1, s2) = siftError(Proj2, p3d, pt1, pt2, 7500);
	//dispProjError(Proj2, pt2, p3d);
	cout << s1.size() << endl;
	Mat Rtvec;
	Rodrigues(Rt, Rtvec, noArray());
	//cout << "tt: " << tt << endl;
	cv::solvePnPRefineVVS(Xd, //p3d
											 s2, //pt2
											 cam,
										 	 noArray(),
											 Rtvec,
											 tt,
											 TermCriteria(TermCriteria::EPS+TermCriteria::COUNT, 25, FLT_EPSILON),1);
 Rtvec.at<double>(1,0) = -Rtvec.at<double>(1,0);
 Rodrigues(Rtvec, Rt, noArray());
 tt = normalizeVec(tt);
 //cout << "tt: " << tt << endl;
 cv::vconcat(Rt,tt.t(),proj);
 Proj2 = cam*proj.t();
 cv::triangulatePoints(Proj1, Proj2, pt1, pt2, p4d);
 p3d = dim4to3(p4d);
 //dispProjError(Proj2, pt2, p3d);
 //cout << endl;
 return {Rt, tt};
/*

	cout << "New rotdiff: " << rotDiff << endl;
	cout << "Old tvec: " << t << endl;
	cout << "New tvec: " << tlm/sqrt(tlm.at<double>(0,0)*tlm.at<double>(0,0)+tlm.at<double>(0,1)*tlm.at<double>(0,1)+tlm.at<double>(0,2)*tlm.at<double>(0,2)) << endl;
	Mat newR;
	Mat Pk = projMat(K,newR,tlm);
	cv::triangulatePoints(Pk_1, Pk, scene1, scene2, pointX); */
}
