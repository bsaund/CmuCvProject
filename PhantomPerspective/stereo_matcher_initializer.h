#pragma once
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/utility.hpp"


enum { STEREO_BM = 0, STEREO_SGBM = 1, STEREO_HH = 2, STEREO_VAR = 3, STEREO_3WAY = 4 };


struct trinsics {
  cv::Mat M1, D1, M2, D2; //intrinsics
  cv::Mat R, T, R1, P1, R2, P2; //extrinsics
  cv::Mat Q;
};

struct maps {
  cv::Mat map11, map12, map21, map22;
};

bool loadCameraParams(std::string intrinsic_filename, std::string extrinsic_filename,
	trinsics &p);

cv::Ptr<cv::StereoBM> getStereoBM(cv::Rect roi1, cv::Rect roi2, int SADWindowSize,
	int numberOfDisparities);

cv::Ptr<cv::StereoSGBM> getStereoSGBM(int SADWindowSize, int numberOfDisparities, int cn, int alg);

