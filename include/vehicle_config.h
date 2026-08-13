#pragma once
#include <string>

// ─────────────────────────────────────────────────────────────────
// 검출 알고리즘(classical / UFLD / 그 외)과 무관하게 공통으로 쓰이는 설정.
// "차선을 어떻게 찾을지"가 아니라 "찾은 차선으로 무엇을 계산할지"에 관한 값이라
// 어떤 모델로 교체하든 이 설정은 그대로 재사용됨.
// ─────────────────────────────────────────────────────────────────
struct VehicleConfig {
  double lane_width_pixels = 400.0;   // lookahead 지점에서의 기준 차선 폭(px)
  double lane_width_meters = 3.5;     // 실제 차선 폭(m) - 도로 표준값
  double lookahead_y_ratio = 0.65;    // offset 계산 기준 y (화면 비율)
  double draw_y_bottom_ratio = 0.95;  // 렌더링 시 차선을 그리는 하단 y (화면 비율)
  double departure_warning_m = 0.5;   // 경고 임계값
  double departure_stop_m = 0.9;      // 위험(정지) 임계값
};

bool load_vehicle_config(const std::string &filepath, VehicleConfig &config);
