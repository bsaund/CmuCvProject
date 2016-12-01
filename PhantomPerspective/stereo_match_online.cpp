/*
*  stereo_match_online.cpp
*
*
*/

//Project1.exe 1 2 -i=../../intrinsics.yml -e=../../extrinsics.yml --blocksize=5 --max-disparity=320 --algorithm=bm

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include "stereo_matcher_initializer.h"
#include "perspective_matcher.h"

#include <stdio.h>

using namespace cv;



static void print_help()
{
	printf("\nDemo stereo matching converting L and R images into disparity and point clouds\n");
	printf("\nUsage: stereo_match <left_image> <right_image> [--algorithm=bm|sgbm|hh|sgbm3way] [--blocksize=<block_size>]\n"
		"[--max-disparity=<max_disparity>] [--scale=scale_factor>] [-i=<intrinsic_filename>] [-e=<extrinsic_filename>]\n"
		"[--no-display] [-o=<disparity_image>] [-p=<point_cloud_file>]\n");
}

static void saveXYZ(const char* filename, const Mat& mat)
{
	const double max_z = 1.0e4;
	FILE* fp = fopen(filename, "wt");
	for (int y = 0; y < mat.rows; y++)
	{
		for (int x = 0; x < mat.cols; x++)
		{
			Vec3f point = mat.at<Vec3f>(y, x);
			if (fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
			fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);
		}
	}
	fclose(fp);
}

int continuousDepthMap(VideoCapture &camL, VideoCapture &camR, Ptr<StereoMatcher> sgbm,
		       int alg, maps &m, trinsics &p) {
	Mat img1, img2, img1_colored, img2_colored;
	char charCheckForEsc = 0;



	while (charCheckForEsc != 27 && camL.isOpened() && camR.isOpened()) {
		namedWindow("left", 1);
		namedWindow("right", 1);
		namedWindow("disparity", 1);
		namedWindow("reprojection",1);
		camL >> img1_colored;
		camR >> img2_colored;


		if (charCheckForEsc == 'c') {
			imwrite("left.jpg", img1);
			imwrite("right.jpg", img2);
		} else {
			img1 = img1_colored.clone();
			img2 = img2_colored.clone();
		}

		if (alg == STEREO_BM) {
			cvtColor(img1, img1, CV_BGR2GRAY);
			cvtColor(img2, img2, CV_BGR2GRAY);
		}

		Mat disp, dispInt, newImage;
		Mat R = Mat::eye(3, 3, cv::DataType<double>::type);
		Mat T = Mat::zeros(3, 1, cv::DataType<double>::type);

		rectifyBoth(img1, img2, m);
		sgbm->compute(img1, img2, disp);

		dispInt.convertTo(disp, CV_32F);
		disp /= 16;  //sgbm returns disp as a 4-fractional-bit short


		getDifferentPerspective(img1_colored, img2_colored,
														R, T, p, disp, newImage);


		imshow("left", img1);
		imshow("right", img2);
		imshow("disparity", disp/sgbm->getNumDisparities());
		imshow("reprojection", newImage);
		charCheckForEsc = cv::waitKey(1);		// delay (in ms) and get key press, if any

	}
	return 1;
}


int mainOnline(int argc, char** argv)
{
	std::string intrinsic_filename = "";
	std::string extrinsic_filename = "";
	std::string disparity_filename = "";
	std::string point_cloud_filename = "";



	int SADWindowSize, numberOfDisparities;
	bool no_display;
	float scale;


	int alg = STEREO_BM;
	
	cv::CommandLineParser parser(argc, argv,
		"{@cam1ind|1|} {@cam2ind|2|}{help h||}{algorithm|bm|}{max-disparity|256|}{blocksize|9|}{no-display||}{scale|1|}{i|intrinsics.yml|}{e|extrinsics.yml|}{o||}{p||}");
	if (parser.has("help"))
	{
		print_help();
		return 0;
	}

	if (parser.has("algorithm"))
	{
		std::string _alg = parser.get<std::string>("algorithm");
		alg = _alg == "bm" ? STEREO_BM :
			_alg == "sgbm" ? STEREO_SGBM :
			_alg == "hh" ? STEREO_HH :
			_alg == "var" ? STEREO_VAR :
			_alg == "sgbm3way" ? STEREO_3WAY : -1;
	}
	numberOfDisparities = parser.get<int>("max-disparity");
	SADWindowSize = parser.get<int>("blocksize");
	scale = parser.get<float>("scale");
	if(scale != 1){
	  printf("Different scale not supported currently");
	  return -1;
	}
	
	no_display = parser.has("no-display");

	intrinsic_filename = parser.get<std::string>("i");
	extrinsic_filename = parser.get<std::string>("e");

	if (parser.has("o"))
		disparity_filename = parser.get<std::string>("o");
	if (parser.has("p"))
		point_cloud_filename = parser.get<std::string>("p");
	if (!parser.check())
	{
		parser.printErrors();
		return 1;
	}
	if (alg < 0)
	{
		printf("Command-line parameter error: Unknown stereo algorithm\n\n");
		print_help();
		return -1;
	}
	if (numberOfDisparities < 1 || numberOfDisparities % 16 != 0)
	{
		printf("Command-line parameter error: The max disparity (--maxdisparity=<...>) must be a positive integer divisible by 16\n");
		print_help();
		return -1;
	}
	if (scale < 0) {
		printf("Command-line parameter error: The scale factor (--scale=<...>) must be a positive floating-point number\n");
		return -1;
	}
	if (SADWindowSize < 1 || SADWindowSize % 2 != 1) {
		printf("Command-line parameter error: The block size (--blocksize=<...>) must be a positive odd number\n");
		return -1;
	}

	if ((intrinsic_filename.empty()) || (extrinsic_filename.empty())) {
		printf("Intrinsics and extrinsics necessary\n");
		return -1;
	}

	if (extrinsic_filename.empty() && !point_cloud_filename.empty())
	{
		printf("Command-line parameter error: extrinsic and intrinsic parameters must be specified to compute the point cloud\n");
		return -1;
	}

	int color_mode = alg == STEREO_BM ? 0 : -1;
	VideoCapture camLeft(parser.get<int>(0));
	VideoCapture camRight(parser.get<int>(1));


	// Mat img1 = imread(img1_filename, color_mode);
	// Mat img2 = imread(img2_filename, color_mode);


	// if (scale != 1.f)
	//   {
	//     Mat temp1, temp2;
	//     int method = scale < 1 ? INTER_AREA : INTER_CUBIC;
	//     resize(img1, temp1, Size(), scale, scale, method);
	//     img1 = temp1;
	//     resize(img2, temp2, Size(), scale, scale, method);
	//     img2 = temp2;
	//   }
	Mat img1;
	camLeft >> img1;

	Size img_size = img1.size();

	Rect roi1, roi2;

	trinsics p;
	if(!loadCameraParams(intrinsic_filename, extrinsic_filename, p)) {
	  printf("Problem loading intrinsics or extrinsics\n");
	  return -1;
	}
	stereoRectify(p.M1, p.D1, p.M2, p.D2, img_size, p.R, p.T, p.R1, p.R2, p.P1, p.P2, p.Q,
		      CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2);

	maps m;
	initUndistortRectifyMap(p.M1, p.D1, p.R1, p.P1, img_size, CV_16SC2, m.map11, m.map12);
	initUndistortRectifyMap(p.M2, p.D2, p.R2, p.P2, img_size, CV_16SC2, m.map21, m.map22);



	int cn = img1.channels();

	//Mat img1p, img2p, dispp;
	//copyMakeBorder(img1, img1p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);
	//copyMakeBorder(img2, img2p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);

	int64 t = getTickCount();

	Ptr<StereoMatcher> usedBm;

	if (alg == STEREO_BM)
		usedBm = getStereoBM(roi1, roi2, SADWindowSize, numberOfDisparities);
	else if (alg == STEREO_SGBM || alg == STEREO_HH || alg == STEREO_3WAY)
	  usedBm = getStereoSGBM(SADWindowSize, numberOfDisparities, cn, alg);

	t = getTickCount() - t;
	printf("Time elapsed: %fms\n", t * 1000 / getTickFrequency());

	//disp = dispp.colRange(numberOfDisparities, img1p.cols);
	// if (alg != STEREO_VAR)
	//   disp.convertTo(disp8, CV_8U, 255 / (numberOfDisparities*16.));
	// else

	if (!no_display) {
	  continuousDepthMap(camLeft, camRight, usedBm, alg, m, p);
	}

	// if (!disparity_filename.empty())
	//   imwrite(disparity_filename, disp8);

	// if (!point_cloud_filename.empty())    {
	//     printf("storing the point cloud...");
	//     fflush(stdout);
	//     Mat xyz;
	//     reprojectImageTo3D(disp, xyz, Q, true);
	//     saveXYZ(point_cloud_filename.c_str(), xyz);
	//     printf("\n");
	//   }


	return 0;
}
