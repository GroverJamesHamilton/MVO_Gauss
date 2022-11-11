#include "function.h"
using namespace cv;
using namespace std;
//
//Brute Force Matcher + displays matches
//
vector<DMatch> BruteForce(Mat img1, Mat img2, vector<KeyPoint> keyp1, vector<KeyPoint> keyp2, Mat desc1, Mat desc2, double ratio) {
	Ptr<BFMatcher> matcher = BFMatcher::create(NORM_HAMMING2, false);
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
Em = findEssentialMat(scene2, scene1, cam, RANSAC, 0.999, 1);
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
Mat E; //Reprojection error vector
double maxError = 1000; //Max repr. error
double X,Y,Z,W, x,y, median, avgY, newScale, e, distance;
	for(int i = 0; i<point3d.cols; i++){
		//Obtain Triangulated 3D-points and their 2D correspondence from last frame
		W = point3d.at<double>(3,i);
		X = point3d.at<double>(0,i)/W;
		Y = point3d.at<double>(1,i)/W;
		Z = point3d.at<double>(2,i)/W;
		x = scene[i].x;
		y = scene[i].y;
		double Xj[4][1] = {X, Y, Z, 1};
		double xj[3][1] = {x, y, 1};
		Mat XJ = Mat(4,1,CV_64F,Xj);
		Mat xJ = Mat(3,1,CV_64F,xj);

		//Calculate repr. error
		E = xJ - PkHat*XJ;
		e = sqrt(E.at<double>(0,0)*E.at<double>(0,0) + E.at<double>(0,1)*E.at<double>(0,1) + E.at<double>(0,2)*E.at<double>(0,2));
		//Some values of Y are close to 1 which are erroneous, therefore removed in condition below (if abs(Y-1) > 0.001)
		if(e < maxError && abs(Y-1) > 0.1 && 1.65/Y > 0.1 && 1.65/Y < 4) //&& height/Y < outlierFactor*prevScale && height/Y > prevScale/outlierFactor
		//if(e < maxError && abs(Y-1) > 0.1 && Y > 0) //&& height/Y < outlierFactor*prevScale && height/Y > prevScale/outlierFactor
		{
			//cout << "Reprojection error: " << e << endl;
			cout << "Ok: " << 1.65/Y << endl;
			//cout << pix << endl;
			//cout << endl;
			Yval.push_back(Y);
		}
		}
size = Yval.size();
cout << "Size: " << size << endl;
sort(Yval.begin(), Yval.end());
for(int i = 0; i < size; i++){
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

// Converts a rotation vector to rotation matrix
Mat eulerAnglesToRotationMatrix(Mat rot)
	{
	    // Calculate rotation about x axis
	    Mat R_x = (Mat_<double>(3,3) <<
	               1,       0,              0,
	               0,       cos(rot.at<int>(0,0)),   -sin(rot.at<int>(0,0)),
	               0,       sin(rot.at<int>(0,0)),   cos(rot.at<int>(0,0))
	               );
	    // Calculate rotation about y axis
	    Mat R_y = (Mat_<double>(3,3) <<
	               cos(rot.at<int>(0,1)),    0,      sin(rot.at<int>(0,1)),
	               0,               1,      0,
	               -sin(rot.at<int>(0,1)),   0,      cos(rot.at<int>(0,1))
	               );
	    // Calculate rotation about z axis
	    Mat R_z = (Mat_<double>(3,3) <<
	               cos(rot.at<int>(0,2)),    -sin(rot.at<int>(0,2)),      0,
	               sin(rot.at<int>(0,2)),    cos(rot.at<int>(0,2)),       0,
	               0,               0,                  1);
	    Mat R = R_z * R_y * R_x;
	    return R;
	}

	vector<size_t> inlierLikelihood(vector<double> yVal, double maxSigma)
	{
		double mu = avg(yVal);
		double var = variance(yVal, mu);
		double sigma = sqrt(var);
		int o = 1;
		/*
		cout << "Overall average: " << mu << endl;
		cout << "Overall variance: " << var << endl;
		cout << "Overall standard deviation: " << sigma << endl;
		*/
		if(sigma > maxSigma)
		{
		arma::mat data(1, yVal.size(), arma::fill::zeros);
		for(int i = 0; i < yVal.size(); i++)
		{
			data(0,i) = yVal[i];
		}
		arma::gmm_diag model;
		bool status3 = model.learn(data, 3, arma::maha_dist, arma::random_subset, 10, 5, 1e-10, false);
		if(status3 == false)
  	{
  		cout << "learning failed" << endl;
  	}
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
			/*
			cout << "New mean: " << model.means[min_idx] << endl;
			cout << "New stdev: " << sqrt(model.dcovs[min_idx]) << endl;
			cout << "Weight: " << model.hefts[min_idx] << endl;
			*/
			//mu = model.means[min_idx];
			//sigma = sqrt(model.dcovs[min_idx]);
			if(abs(mean0 - mean1) <= abs(mean1 - mean2))
			if(o)
			{
				bool status2 = model.learn(data, 2, arma::maha_dist, arma::random_subset, 10, 5, 1e-10, false);
				if(status2)
				{
					means.clear();
					double mean0 = model.means(0);
					double mean1 = model.means(1);
					means.push_back(mean0);
					means.push_back(mean1);
					auto it = minmax_element(begin(means), end(means));
					int min_idx = std::distance(means.begin(), it.first);
					/*
					//imp
					cout << endl;
					cout << "New mean: " << model.means[min_idx] << endl;
					cout << "New stdev: " << sqrt(model.dcovs[min_idx]) << endl;
					cout << "Weight: " << model.hefts[min_idx] << endl;
					cout << endl;
					*/
					//mu = model.means[min_idx];
					//sigma = sqrt(model.dcovs[min_idx]);
				}
			}
		}
	}
	double lh;
	vector<double> likelihoods;
	double pi = 3.14159;
	for(int i = 0; i < yVal.size(); i++)
	{
		lh = 1/(sqrt(2*pi)*sigma)*exp(-pow((yVal[i] - mu)/sigma, 2)/2);
		likelihoods.push_back(lh);
		//cout << "Y: " << yVal[i] << " mu: " << lh << endl;
	}

	vector<size_t> idx(yVal.size());
	iota(idx.begin(), idx.end(), 0);
	stable_sort(idx.begin(), idx.end(), [&likelihoods](size_t i1, size_t i2) {return likelihoods[i1] > likelihoods[i2];});
	return idx;
}
// Sifts out abnormaly large triangulated 3D-points and their 2D-point correspondences
tuple <vector<Point3d>,vector<Point2d>,vector<Point2d>> siftPoints(Mat X3D, Mat proj, vector<Point2d> scene1, vector<Point2d> scene2, int maxPoints, int maxError, bool show)
{
vector<double> yVal, erep;
vector<Point2d> sift1, sift2, sort1, sort2;
vector<Point3d> coords, coordsSifted;
double Xs,Ys,Zs, e;
int j = 0;
Mat Em;
Mat Xk = cv::Mat::ones(cv::Size(1,4), CV_64F);
Mat xk = cv::Mat::ones(cv::Size(1,3), CV_64F);
for(int i = 0; i < X3D.cols; i++)
{
		Xs = X3D.at<double>(0,i)/X3D.at<double>(3,i);
		Ys = X3D.at<double>(1,i)/X3D.at<double>(3,i);
		Zs = X3D.at<double>(2,i)/X3D.at<double>(3,i);
		Xk.at<double>(0,0) = Xs;
		Xk.at<double>(0,1) = Ys;
		Xk.at<double>(0,2) = Zs;
		// Form the current Image-point from input with index i
		xk.at<double>(0,0) = scene2[i].x;
		xk.at<double>(0,1) = scene2[i].y;
		Em = xk - proj*Xk;
		e = sqrt(pow(Em.at<double>(0,0), 2) + pow(Em.at<double>(0,1), 2) + pow(Em.at<double>(0,2), 2));
		if(e < maxError)// 3D-points can't be behind the camera
		{
		if(show)
		{
			//cout << e << endl;
			//cout << scene1[i] << scene2[i] << endl;
			//cout << Xs << "," << Ys << "," << Zs << ";" << endl;
		}
		coords.push_back({Xs,Ys,Zs});
		sift1.push_back(scene1[i]);
		sift2.push_back(scene2[i]);
		yVal.push_back(Ys);
		erep.push_back(e);
	  }
}
vector<size_t> index = inlierLikelihood(yVal, 0.15);
//imp
/*
viz::Viz3d window("3D-coords");
window.showWidget("coordinate", viz::WCoordinateSystem());
window.showWidget("points", viz::WCloud(coords, viz::Color::green()));
window.spin();
*/

//vector<size_t> index(yVal.size());
//iota(index.begin(), index.end(), 0);
//stable_sort(index.begin(), index.end(), [&erep](size_t i1, size_t i2) {return erep[i1] < erep[i2];});

//cout << "The sifted and sorted based on repr error" << endl;

if(maxPoints > index.size()){ maxPoints = index.size(); }

for(int n = 0; n < maxPoints; n++)
{
	if(n == yVal.size())
	{
		break;
	}
	//imp
	//cout << coords[index[n]] << endl;
	sort1.push_back(sift1[index[n]]);
	sort2.push_back(sift2[index[n]]);
	coordsSifted.push_back(coords[index[n]]);
	/*
	cout << coords[n] << endl;
	sort1.push_back(sift1[n]);
	sort2.push_back(sift2[n]);
	coordsSifted.push_back(coords[n]);
	if(show)
	*/
	{
		//cout << eRepr[idx[n]] << endl;
		//cout << coords[idx[n]] << endl;
	}
}

return{coordsSifted, sort1, sort2};
}
// Returns the index of the triangulated 3D-point with the lowest reprojection error
// Not currently used in the algorithm
int poseRefiner(Mat cam, Mat proj, vector<Point3d> Xtriang, vector<Point2d> scene1, vector<Point2d> scene2)
{
// Repr.error = |xk - Xk*P|
Mat Xk = cv::Mat::ones(cv::Size(1,4), CV_64F); //
Mat xk = cv::Mat::ones(cv::Size(1,3), CV_64F);
Mat Er; // Reprojection error matrix xk - Xk*P
double e; // Resulting reprojection error
double min = 1000000; // Start with relative high value to compare repr.errors with
int index;
for (size_t i = 0; i < Xtriang.size(); i++)
{
// Form the current 3D-point from input with index i
Xk.at<double>(0,0) = Xtriang[i].x;
Xk.at<double>(0,1) = Xtriang[i].y;
Xk.at<double>(0,2) = Xtriang[i].z;
// Form the current Image-point from input with index i
xk.at<double>(0,0) = scene2[i].x;
xk.at<double>(0,1) = scene2[i].y;

Er = xk - proj*Xk;
e = sqrt(Er.at<double>(0,0)*Er.at<double>(0,0) + Er.at<double>(0,1)*Er.at<double>(0,1) + Er.at<double>(0,2)*Er.at<double>(0,2));

if (e < min)
{
	min = e;
	index = i;
}
}
return index;
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
	cout << 1.65/average << ", " ;
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
cout << 1.65/heights[index] << ", " ;
return {heights[index], index};
}

tuple <double, double, double, double> generateHeights(vector<Point3d> X, double realH, double maxScale)
{
vector<Point3d> normal;
Mat point, N, N_norm;
Mat n = cv::Mat::zeros(3,3,CV_64F);
Mat l = cv::Mat::ones(3,1,CV_64F);
double norm, height, n1, n2, n3;
vector<double> heights;
vector <vector <int>> bin;
bin = combList(X.size(), 3); //Combination list of sizeof(X) choose 3 numbers
if(X.size() > 2)
{
int u = 0;
for(int i = 0; i < bin.size(); i++)
{
for(int j = 0; j < 3; j++)
{
	n.at<double>(j,0) = X[bin[i][j]].x;
	n.at<double>(j,1) = X[bin[i][j]].y;
	n.at<double>(j,2) = X[bin[i][j]].z;
}
N = n.inv()*l;
norm = sqrt(pow(N.at<double>(0,0), 2) + pow(N.at<double>(1,0), 2) + pow(N.at<double>(2,0), 2));
N_norm = N/norm;
height = 1/norm;
n1 = N_norm.at<double>(0,0);
n2 = N_norm.at<double>(1,0);
n3 = N_norm.at<double>(2,0);
//imp
//if(height < 2 && height > 1.6)// && n1 > 0 && n3 > 0)
if(n2 < -0.95)
//if(n1*X[list[0]].x + n3*X[list[0]].z > 0 && n2 < -0.99)
{
normal.push_back({n1,n2,n3});
heights.push_back(height);

//cout << bin[i][0] << " " << bin[i][1] << " " << bin[i][2] << endl;
u++;
/*
cout << "It: " << u << endl;
cout << "Height est: " << height << endl;
cout << "Normal vector: " << N_norm << endl;
cout << endl;
*/
}
}
}
if(heights.size() > 5)
{
double average = avg(heights);
int index;
double h;
tie(h,index) = qScore(heights);
//cout << "Best normal estimate: " << normal[index] << endl;
sort(heights.begin(), heights.end());
for(int k = 0; k < heights.size(); k++)
{
	//cout << heights[k] << endl;
}
//cout << "Size: " << heights.size() << endl;
//cout << heights[index] << endl;
medianAvg(heights);
//cout << "Average: " << average << endl;
}
return {1,1,1,1};
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

void nelderMead(vector<Point3d> normalVec, double h, Mat cam, vector<Point2d> scene1,vector<Point2d> scene2, Mat t, Mat rota)
{
	double n,n1,n2,n3,norm;
	Mat H;
	double nDiff = 0.4;
	double hDiff = 2;
	int iter = 20;
	double p = 1.5;
	double SAD;
	double min = 100000;
	double hbest;
	Mat x1 = cv::Mat::ones(cv::Size(1,3), CV_64F);
	Mat x2 = cv::Mat::ones(cv::Size(1,3), CV_64F);
	Mat err = cv::Mat::ones(cv::Size(1,3), CV_64F);
	Mat N = cv::Mat::zeros(cv::Size(3,1), CV_64F);
	n1 = normalVec[0].x;
	n2 = normalVec[0].y;
	n3 = normalVec[0].z;
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
			H = cam*(rota + t*N/h)*cam.inv();
			SAD = 0;
			for(int k = 0; k < scene1.size(); k++)
			{
				x1.at<double>(0,0) = scene1[k].x;
				x1.at<double>(1,0) = scene1[k].y;
				x2.at<double>(0,0) = scene2[k].x;
				x2.at<double>(1,0) = scene2[k].y;
				err = x2 - H*x1;
				auto e = sqrt(pow(err.at<double>(0,0),2) + pow(err.at<double>(0,1),2) + pow(err.at<double>(0,1),2));
				SAD = SAD + e;
			}
			if(SAD < min)
			{
				hbest = h;
				min = SAD;
				auto norm = sqrt(pow(n1,2)+pow(n2,2)+pow(n,2));
				cout << "New best estimate: " << hbest << " i: " << i << " j: " << j << " Norm: " << norm << " SAD: " << SAD << endl;
			}
		}
	}
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
	tie(Xd, s1, s2) = siftPoints(p4d, Proj2, pt1, pt2, 50, 10000, true);
	//bork
	p3d = dim4to3(p4d);
	dispProjError(Proj2, pt2, p3d);
	Mat Rtvec;
	Rodrigues(Rt, Rtvec, noArray());
	cv::solvePnPRefineLM(Xd, //p3d
											 s2, //pt2
											 cam,
										 	 noArray(),
											 Rtvec,
											 tt,
											 TermCriteria(TermCriteria::EPS+TermCriteria::COUNT, 20, FLT_EPSILON));
 Rodrigues(Rtvec, Rt, noArray());
 tt = normalizeVec(tt);
 //cout << "tt: " << tt << endl;
 cv::vconcat(Rt,tt.t(),proj);
 Proj2 = cam*proj.t();
 cv::triangulatePoints(Proj1, Proj2, pt1, pt2, p4d);
 p3d = dim4to3(p4d);
 dispProjError(Proj2, pt2, p3d);
 return {Rt, tt};
/*

	cout << "New rotdiff: " << rotDiff << endl;
	cout << "Old tvec: " << t << endl;
	cout << "New tvec: " << tlm/sqrt(tlm.at<double>(0,0)*tlm.at<double>(0,0)+tlm.at<double>(0,1)*tlm.at<double>(0,1)+tlm.at<double>(0,2)*tlm.at<double>(0,2)) << endl;
	Mat newR;
	Mat Pk = projMat(K,newR,tlm);
	cv::triangulatePoints(Pk_1, Pk, scene1, scene2, pointX); */
}
