#include "perspective_matcher.h"

using namespace cv;

void getDifferentPerspective(Mat &img1, Mat &img2, Mat &R, Mat &T,
							 Ptr<StereoMatcher> sgbm,
							 maps &m, trinsics &p, Mat &disp, 
							 Mat &newImage) {

	Mat img1r, img2r;
	
	remap(img1, img1r, m.map11, m.map12, INTER_LINEAR);
	remap(img2, img2r, m.map21, m.map22, INTER_LINEAR);
	
	img1 = img1r;
	img2 = img2r;

	sgbm->compute(img1, img2, disp);

	Mat _3dImage;
	reprojectImageTo3D(disp, _3dImage, p.Q, true);
	Mat imagePoints;


	_3dImage = _3dImage.reshape(3, 1);
	// projectPoints(_3dImage, Mat::eye(3, 3, cv::DataType<double>::type),
	// 	Mat::zeros(3, 1, cv::DataType<double>::type), p.M1, Mat(),
	// 	imagePoints);
	projectPoints(_3dImage, R, T, p.M1, Mat(), imagePoints);
		
	//printf("rows: %d, cols: %d, dims: %d, ", _3dImage.rows, _3dImage.cols, _3dImage.dims);
	//printf("channels: %d\n", _3dImage.channels());
	//printf("rows: %d, cols: %d, dims: %d, chan: %d", imagePoints.rows, imagePoints.cols, imagePoints.dims, imagePoints.channels());

	newImage = Mat::zeros(img1.size(), CV_8U);
	for (int i = 0; i < imagePoints.rows; i++) {
		//printf("depth: %f\n", _3dImage.at<Vec3f>(0,i)[2]);
		if (_3dImage.at<Vec3f>(0, i)[2] > 100)
			continue;

		Vec2f v = imagePoints.at<Vec2f>(i, 0);
		int x = round(v[0]);
		int y = round(v[1]);

		if (x < 0 || y < 0 || y >= newImage.rows || x >= newImage.cols)
			continue;
		newImage.at<uchar>(y, x) = img1.at<uchar>(i);
	}
}


