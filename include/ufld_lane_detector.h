#pragma once
#include "lane_detector.h"
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <string>
#include <vector>

// Ultra-Fast-Lane-Detection(CULane 사전학습) ONNX 모델을 onnxruntime C++ API로
// 돌리는 검출기.
//
// ⚠️ 아래 상수들(kInputW/H, kGridingNum, kClsNumPerLane, kNumLanes, kRowAnchor)은
//    "표준 UFLD v1 + CULane" 기준값입니다. 실제로 갖고 계신 .onnx 파일이
//    - TuSimple로 학습된 것이거나
//    - UFLDv2/다른 backbone이거나
//    - 다른 입력 해상도로 export된 것이면
//    반드시 이 값들을 그 모델의 config에 맞게 바꿔야 정상 동작합니다.
//    (원본 저장소의 configs/culane_res18.py 등에서 확인 가능)
class UFLDLaneDetector : public LaneDetector {
public:
  // temporal_alpha: 이전 프레임과 섞는 비율(낮을수록 부드럽지만 반응 느림, 기본 0.3)
  // minimum_confidence: 이 이상이어야 화면에 선을 그림 (기본 0.3)
  // max_coast_frames: 검출이 잠깐 끊겨도 이전 궤적을 유지하는 최대 프레임 수 (기본 8)
  // max_track_deviation_px: 이전 추적 위치에서 이 값(px) 이상 튀면 이상치로 버림 (기본 180px)
  explicit UFLDLaneDetector(const std::string &onnx_model_path,
                             double temporal_alpha = 0.3,
                             double minimum_confidence = 0.3,
                             int max_coast_frames = 20,
                             double max_track_deviation_px = 180.0);
  ~UFLDLaneDetector() override;

  LaneResult detect(const cv::Mat &frame) override;
  std::string name() const override { return "UFLD (Ultra-Fast-Lane-Detection, CULane)"; }

private:
  Ort::Env env_;
  Ort::Session session_;
  Ort::MemoryInfo memory_info_;
  std::string input_name_;
  std::string output_name_;

  // ── temporal smoothing 상태 (classical_lane_detector.cpp와 동일한 방식) ──
  double temporal_alpha_;
  double minimum_confidence_;
  int max_coast_frames_;
  double max_track_deviation_px_;

  double smoothed_left_slope_ = 0, smoothed_left_intercept_ = 0, conf_left_ = 0;
  double smoothed_right_slope_ = 0, smoothed_right_intercept_ = 0, conf_right_ = 0;
  int left_coast_count_ = 0, right_coast_count_ = 0;
  bool initialized_ = false;

  // 이번 프레임만의 raw 검출 결과 (smoothing 적용 전)
  struct RawDetection {
    bool left_valid = false, right_valid = false;
    double left_slope = 0, left_intercept = 0;
    double right_slope = 0, right_intercept = 0;
  };

  // 입력 전처리: BGR 프레임 -> 정규화된 CHW float tensor
  std::vector<float> preprocess(const cv::Mat &frame);

  // 출력 후처리: 모델 raw output -> 이번 프레임만의 raw 좌/우 lane fit
  RawDetection postprocess(const std::vector<float> &output, int frame_w, int frame_h);
};
