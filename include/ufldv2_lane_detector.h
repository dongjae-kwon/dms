#pragma once
#include "lane_detector.h"
#include <onnxruntime_cxx_api.h>
#include <array>
#include <memory>
#include <string>
#include <vector>

// Ultra-Fast-Lane-Detection-v2 (CULane res18, 320x1600) ONNX 추론 검출기.
//
// v1과의 핵심 차이:
//  - "차선 없음"을 그리드 클래스 안에 끼워넣지 않고 exist_row/exist_col라는
//    별도 출력(존재 여부 이진분류)으로 분리
//  - row anchor(가로 방향)뿐 아니라 col anchor(세로 방향)까지 같이 예측해서
//    급커브 구간에서도 더 안정적으로 잡도록 설계됨
//
// ⚠️ 아래 상수들은 Netron으로 확인한 그래프(loc_row 57600 + exist_row 576 +
//    loc_col 32400 + exist_col 648 = 91224)로부터 역산한 CULane res18 표준
//    설정값입니다. res34나 다른 데이터셋(tusimple/curvelanes) 모델을 쓰시면
//    이 값들이 달라지니 주의하세요.
class UFLDv2LaneDetector : public LaneDetector {
public:
  explicit UFLDv2LaneDetector(const std::string &onnx_model_path,
                               double temporal_alpha = 0.3,
                               double minimum_confidence = 0.3,
                               int max_coast_frames = 20,
                               double max_track_deviation_px = 180.0);
  ~UFLDv2LaneDetector() override;

  LaneResult detect(const cv::Mat &frame) override;
  std::string name() const override { return "UFLDv2 (CurveLanes res18)"; }

private:
  Ort::Env env_;
  Ort::Session session_;
  Ort::MemoryInfo memory_info_;
  std::string input_name_;
  std::vector<std::string> output_names_; // ["loc_row","exist_row","loc_col","exist_col"] 순서 확인 후 채움

  double temporal_alpha_;
  double minimum_confidence_;
  int max_coast_frames_;
  double max_track_deviation_px_;

  double smoothed_left_slope_ = 0, smoothed_left_intercept_ = 0, conf_left_ = 0;
  double smoothed_right_slope_ = 0, smoothed_right_intercept_ = 0, conf_right_ = 0;
  int left_coast_count_ = 0, right_coast_count_ = 0;
  bool initialized_ = false;
  int debug_dump_frames_left_ = 2; // 진단용: 처음 2프레임만 상세 로그 출력

  struct RawDetection {
    bool left_valid = false, right_valid = false;
    double left_slope = 0, left_intercept = 0;
    double right_slope = 0, right_intercept = 0;
  };

  std::vector<float> preprocess(const cv::Mat &frame);
  RawDetection postprocess(const float *loc_row, const float *exist_row,
                            const float *loc_col, const float *exist_col,
                            int frame_w, int frame_h);
};
