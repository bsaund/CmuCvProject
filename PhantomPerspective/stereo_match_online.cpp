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

#include <stdio.h>

using namespace cv;
enum { STEREO_BM = 0, STEREO_SGBM = 1, STEREO_HH = 2, STEREO_VAR = 3, STEREO_3WAY = 4 };

int alg = STEREO_SGBM;
Mat map11, map12, map21, map22;

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

int continuousDepthMap(VideoCapture &camL, VideoCapture &camR, Ptr<StereoMatcher> sgbm) {
	Mat img1, img2;
	char charCheckForEsc = 0;



	while (charCheckForEsc != 27 && camL.isOpened() && camR.isOpened()) {
		namedWindow("left", 1);
		namedWindow("right", 1);
		namedWindow("disparity", 0);
		camL >> img1;
		camR >> img2;

		if (charCheckForEsc == 'c') {
			imwrite("left.jpg", img1);
			imwrite("right.jpg", img2);
		}

		Mat img1r, img2r;
		Mat disp, disp8;
		remap(img1, img1r, map11, map12, INTER_LINEAR);
		remap(img2, img2r, map21, map22, INTER_LINEAR);

		img1 = img1r;
		img2 = img2r;

		if (alg == STEREO_BM) {
			cvtColor(img1, img1, CV_BGR2GRAY);
			cvtColor(img2, img2, CV_BGR2GRAY);
		}


		sgbm->compute(img1, img2, disp);
		disp /= 10;
		disp.convertTo(disp8, CV_8U);


		imshow("left", img1);
		imshow("right", img2);
		imshow("disparity", disp8);
		charCheckForEsc = cv::waitKey(1);		// delay (in ms) and get key press, if any

	}
	return 1;
}


int main(int argc, char** argv)
{
	std::string intrinsic_filename = "";
	std::string extrinsic_filename = "";
	std::string disparity_filename = "";
	std::string point_cloud_filename = "";



	int SADWindowSize, numberOfDisparities;
	bool no_display;
	float scale;


	Ptr<StereoBM> bm = StereoBM::create(16, 9);
	Ptr<StereoSGBM> sgbm = StereoSGBM::create(0, 16, 3);
	cv::CommandLineParser parser(argc, argv,
		"{@cam1ind|1|} {@cam2ind|2|}{help h||}{algorithm||}{max-disparity|0|}{blocksize|0|}{no-display||}{scale|1|}{i|intrinsics.yml|}{e|extrinsics.yml|}{o||}{p||}");
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
	Mat Q;

	if (!intrinsic_filename.empty()) {
		// reading intrinsic parameters
		FileStorage fs(intrinsic_filename, FileStorage::READ);
		if (!fs.isOpened())
		{
			printf("Failed to open file %s\n", intrinsic_filename.c_str());
			return -1;
		}

		Mat M1, D1, M2, D2;
		fs["M1"] >> M1;
		fs["D1"] >> D1;
		fs["M2"] >> M2;
		fs["D2"] >> D2;

		M1 *= scale;
		M2 *= scale;

		fs.open(extrinsic_filename, FileStorage::READ);
		if (!fs.isOpened())
		{
			printf("Failed to open file %s\n", extrinsic_filename.c_str());
			return -1;
		}

		Mat R, T, R1, P1, R2, P2;
		fs["R"] >> R;
		fs["T"] >> T;

		stereoRectify(M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2);


		initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
		initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);

	}

	numberOfDisparities = numberOfDisparities > 0 ? numberOfDisparities : ((img_size.width / 8) + 15) & -16;

	bm->setROI1(roi1);
	bm->setROI2(roi2);
	bm->setPreFilterCap(31);
	bm->setBlockSize(SADWindowSize > 0 ? SADWindowSize : 9);
	bm->setMinDisparity(0);
	bm->setNumDisparities(numberOfDisparities);
	bm->setTextureThreshold(10);
	bm->setUniquenessRatio(15);
	bm->setSpeckleWindowSize(100);
	bm->setSpeckleRange(32);
	bm->setDisp12MaxDiff(1);

	sgbm->setPreFilterCap(63);
	int sgbmWinSize = SADWindowSize > 0 ? SADWindowSize : 3;
	sgbm->setBlockSize(sgbmWinSize);

	int cn = img1.channels();

	sgbm->setP1(8 * cn*sgbmWinSize*sgbmWinSize);
	sgbm->setP2(32 * cn*sgbmWinSize*sgbmWinSize);
	sgbm->setMinDisparity(0);
	sgbm->setNumDisparities(numberOfDisparities);
	sgbm->setUniquenessRatio(10);
	sgbm->setSpeckleWindowSize(100);
	sgbm->setSpeckleRange(32);
	sgbm->setDisp12MaxDiff(1);
	if (alg == STEREO_HH)
		sgbm->setMode(StereoSGBM::MODE_HH);
	else if (alg == STEREO_SGBM)
		sgbm->setMode(StereoSGBM::MODE_SGBM);
	else if (alg == STEREO_3WAY)
		sgbm->setMode(StereoSGBM::MODE_SGBM_3WAY);


	//Mat img1p, img2p, dispp;
	//copyMakeBorder(img1, img1p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);
	//copyMakeBorder(img2, img2p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);

	int64 t = getTickCount();

	Ptr<StereoMatcher> usedBm;

	if (alg == STEREO_BM)
		usedBm = bm;
	else if (alg == STEREO_SGBM || alg == STEREO_HH || alg == STEREO_3WAY)
		usedBm = sgbm;

	t = getTickCount() - t;
	printf("Time elapsed: %fms\n", t * 1000 / getTickFrequency());

	//disp = dispp.colRange(numberOfDisparities, img1p.cols);
	// if (alg != STEREO_VAR)
	//   disp.convertTo(disp8, CV_8U, 255 / (numberOfDisparities*16.));
	// else

	if (!no_display) {
		continuousDepthMap(camLeft, camRight, usedBm);
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
