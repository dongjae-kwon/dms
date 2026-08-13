#pragma once
#include <opencv2/opencv.hpp>
#include <string>

// ─────────────────────────────────────────────────────────────────
// 검출 결과 (모든 검출기가 공통으로 반환하는 형식)
// 기존 코드 컨벤션 유지: 직선은 y = slope * x + intercept 형태로 표현하고,
// x = (y - intercept) / slope 로 특정 y에서의 x 좌표를 구함 (calc_x_at_y와 동일)
// ─────────────────────────────────────────────────────────────────
struct LaneResult {
  bool left_valid = false;
  bool right_valid = false;

  double left_slope = 0.0;
  double left_intercept = 0.0;
  double right_slope = 0.0;
  double right_intercept = 0.0;

  double conf_left = 0.0;   // 0.0 ~ 1.0
  double conf_right = 0.0;  // 0.0 ~ 1.0

  // 진단용: 이번 프레임에 모델/알고리즘이 "원래" 찾았는지 (이상치 필터링 이전).
  // false인데 이게 자주 false면 모델 인식 성능 문제, true인데 최종 valid가
  // false면 이상치 필터가 너무 빡빡한 것.
  bool raw_left_valid = false;
  bool raw_right_valid = false;

  std::string status_label; // OSD 표시용 (예: "MODE: SUNNY (Auto)", "UFLD (CULane)")
};

// 차선 검출 알고리즘/모델의 공통 인터페이스.
// main 루프는 이 인터페이스만 알면 되고, 내부가 고전 CV인지 딥러닝 모델인지는 몰라도 됨.
// 새 모델(CLRNet, LaneATT 등)을 테스트하고 싶으면 이 인터페이스를 구현하는
// 클래스 하나만 추가하면 되고, main.cpp / CMakeLists.txt의 실행 옵션만 늘리면 됨.
class LaneDetector {
public:
  virtual ~LaneDetector() = default;

  // 한 프레임을 받아 차선 검출 결과를 반환. 내부적으로 이전 프레임 상태를
  // 들고 있어도 됨 (temporal smoothing 등은 각 구현체 책임).
  virtual LaneResult detect(const cv::Mat &frame) = 0;

  // OSD/로그에 표시할 검출기 이름
  virtual std::string name() const = 0;
};
