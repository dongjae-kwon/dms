#pragma once
#include "lane_detector.h"
#include <memory>
#include <string>

// 기존 main.cpp에 있던 "날씨별 config 자동전환 + HSV/Canny/Hough + temporal
// smoothing" 알고리즘 전체를 그대로 담고 있는 검출기. 세부 구현(LaneConfig,
// LightMode, 코스팅 카운터 등)은 .cpp의 Impl 안에 숨겨서 헤더가 깔끔하게 유지됨.
class ClassicalLaneDetector : public LaneDetector {
public:
  // 4가지 날씨별 config 파일 경로를 받아 초기화 (기존 main.cpp의 argv와 동일)
  ClassicalLaneDetector(const std::string &sunny_path,
                         const std::string &overcast_path,
                         const std::string &rainy_path,
                         const std::string &night_path);
  ~ClassicalLaneDetector() override;

  LaneResult detect(const cv::Mat &frame) override;
  std::string name() const override;

  // 디버그용: 가장 최근 detect() 호출에서 쓰인 색상 마스크(White+Yellow) +
  // ROI 오버레이를 반환. 왜 특정 방향(특히 우측) 차선이 안 잡히는지 눈으로
  // 확인할 때 씀. detect()를 한 번도 안 불렀으면 빈 Mat 반환.
  cv::Mat debug_mask_overlay() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
