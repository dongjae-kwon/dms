#pragma once
#include "lane_detector.h"
#include "vehicle_config.h"
#include <opencv2/opencv.hpp>

// LaneResult + VehicleConfig만으로 offset을 계산하고 화면에 그려주는 함수.
// 검출기가 classical이든 UFLD든 동일하게 동작함 (검출기 교체가 렌더링에
// 영향을 주지 않도록 분리).
cv::Mat render_frame(const cv::Mat &frame, const LaneResult &result,
                      const VehicleConfig &vcfg, double &out_offset_m);
