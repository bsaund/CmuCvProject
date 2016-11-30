#pragma once
#include "stereo_matcher_initializer.h"

void rectifyBoth(cv::Mat &img1, cv::Mat &img2, const maps &m);

void getDifferentPerspective(cv::Mat img1_colored, cv::Mat img2_colored, 
														 cv::Mat &R, cv::Mat &T,
														 trinsics &p, 
														 cv::Mat &disp, cv::Mat &newImage);
