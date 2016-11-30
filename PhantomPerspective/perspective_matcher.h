#pragma once
#include "stereo_matcher_initializer.h"

void getDifferentPerspective(cv::Mat &img1, cv::Mat &img2, 
														 cv::Mat img1_colored, cv::Mat img2_colored, 
														 cv::Mat &R, cv::Mat &T,
														 cv::Ptr<cv::StereoMatcher> sgbm,
														 maps &m, trinsics &p, 
														 cv::Mat &disp, cv::Mat &newImage);
