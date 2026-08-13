#include "classical_lane_detector.h"
#include <cjson/cJSON.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace {

// ── 원본 main.cpp의 LaneConfig에서 offset/경고 계산용 필드는 제거하고
//    "알고리즘 튜닝 파라미터"만 남김 (그 값들은 이제 VehicleConfig가 담당) ──
struct LaneConfig {
  double roi_top_ratio, roi_bottom_ratio, roi_left_ratio, roi_right_ratio;
  int white_hsv_lower[3], white_hsv_upper[3];
  int yellow_hsv_lower[3], yellow_hsv_upper[3];
  int gaussian_kernel_size;
  double canny_low_threshold, canny_high_threshold;
  double hough_rho, hough_theta_deg;
  int hough_threshold;
  double hough_min_line_length, hough_max_line_gap;
  double minimum_absolute_slope, maximum_absolute_slope;
  double temporal_alpha;
  double minimum_confidence;
  double max_track_deviation_px = 80.0; // 이전 프레임 위치 기준 이상치 제거 임계값 (기본 80px)
  double max_color_blob_area_px = 6000.0; // 이보다 큰 흰색/노란색 덩어리는 노면눈부심 등으로 간주하고 제거
  // 차선 폭 일관성 필터: 기준 폭 대비 허용 오차 비율 (0.20 = ±20%)
  // 합류/분기 구간에서 새 차선으로 트래커가 점진적으로 튀는 것을 방지
  double lane_width_tolerance_ratio = 0.20;
};

double calc_x_at_y(double slope, double intercept, double y) {
  if (fabs(slope) < 1e-5) return 0.0;
  return (y - intercept) / slope;
}

struct LineSegment {
  double slope = 0, intercept = 0, length = 0;
  bool valid = false;
};

enum LightMode { MODE_SUNNY, MODE_OVERCAST, MODE_RAINY, MODE_NIGHT };

constexpr double BRIGHTNESS_NIGHT_ENTER = 50.0;
constexpr double BRIGHTNESS_NIGHT_EXIT = 70.0;
constexpr double BRIGHTNESS_RAINY_ENTER = 95.0;
constexpr double BRIGHTNESS_SUNNY_ENTER = 135.0;
constexpr double SATURATION_OVERCAST_MAX = 35.0;
constexpr double SATURATION_SUNNY_MIN = 35.0;
constexpr int STABLE_FRAME_THRESHOLD = 10;
constexpr int MAX_COAST_FRAMES = 8;

bool load_config(const char *filepath, LaneConfig *config) {
  FILE *file = fopen(filepath, "r");
  if (!file)
    return false;
  fseek(file, 0, SEEK_END);
  long length = ftell(file);
  fseek(file, 0, SEEK_SET);
  char *buffer = (char *)malloc(length + 1);
  size_t bytes_read = fread(buffer, 1, length, file);
  if (bytes_read != (size_t)length) 
  {
    fclose(file);
    return false;
  }
  buffer[length] = '\0';
  fclose(file);
  cJSON *json = cJSON_Parse(buffer);
  free(buffer);
  if (!json)
    return false;

  config->roi_top_ratio = cJSON_GetObjectItem(json, "roi_top_ratio")->valuedouble;
  config->roi_bottom_ratio = cJSON_GetObjectItem(json, "roi_bottom_ratio")->valuedouble;
  config->roi_left_ratio = cJSON_GetObjectItem(json, "roi_left_ratio")->valuedouble;
  config->roi_right_ratio = cJSON_GetObjectItem(json, "roi_right_ratio")->valuedouble;

  cJSON *w_low = cJSON_GetObjectItem(json, "white_hsv_lower");
  cJSON *w_upp = cJSON_GetObjectItem(json, "white_hsv_upper");
  cJSON *y_low = cJSON_GetObjectItem(json, "yellow_hsv_lower");
  cJSON *y_upp = cJSON_GetObjectItem(json, "yellow_hsv_upper");
  for (int i = 0; i < 3; i++) {
    config->white_hsv_lower[i] = cJSON_GetArrayItem(w_low, i)->valueint;
    config->white_hsv_upper[i] = cJSON_GetArrayItem(w_upp, i)->valueint;
    config->yellow_hsv_lower[i] = cJSON_GetArrayItem(y_low, i)->valueint;
    config->yellow_hsv_upper[i] = cJSON_GetArrayItem(y_upp, i)->valueint;
  }

  config->gaussian_kernel_size = cJSON_GetObjectItem(json, "gaussian_kernel_size")->valueint;
  config->canny_low_threshold = cJSON_GetObjectItem(json, "canny_low_threshold")->valuedouble;
  config->canny_high_threshold = cJSON_GetObjectItem(json, "canny_high_threshold")->valuedouble;
  config->hough_rho = cJSON_GetObjectItem(json, "hough_rho")->valuedouble;
  config->hough_theta_deg = cJSON_GetObjectItem(json, "hough_theta_deg")->valuedouble;
  config->hough_threshold = cJSON_GetObjectItem(json, "hough_threshold")->valueint;
  config->hough_min_line_length = cJSON_GetObjectItem(json, "hough_min_line_length")->valuedouble;
  config->hough_max_line_gap = cJSON_GetObjectItem(json, "hough_max_line_gap")->valuedouble;
  config->minimum_absolute_slope = cJSON_GetObjectItem(json, "minimum_absolute_slope")->valuedouble;
  config->maximum_absolute_slope = cJSON_GetObjectItem(json, "maximum_absolute_slope")->valuedouble;
  config->temporal_alpha = cJSON_GetObjectItem(json, "temporal_alpha")->valuedouble;
  config->minimum_confidence = cJSON_GetObjectItem(json, "minimum_confidence")->valuedouble;

  // 새 필드: 없으면 구조체 기본값(80px) 그대로 사용 - 기존 JSON 파일 그대로 둬도 안 깨짐
  cJSON *track_dev = cJSON_GetObjectItem(json, "max_track_deviation_px");
  if (track_dev && cJSON_IsNumber(track_dev))
    config->max_track_deviation_px = track_dev->valuedouble;

  // 차선 폭 일관성 필터 허용 오차 (없으면 기본값 0.20 유지)
  cJSON *width_tol = cJSON_GetObjectItem(json, "lane_width_tolerance_ratio");
  if (width_tol && cJSON_IsNumber(width_tol))
    config->lane_width_tolerance_ratio = width_tol->valuedouble;

  cJSON_Delete(json);
  return true;
}

cv::Mat create_roi_mask(int width, int height, const LaneConfig *config) {
  cv::Mat mask = cv::Mat::zeros(height, width, CV_8UC1);
  std::vector<cv::Point> pts = {
      cv::Point(width * config->roi_left_ratio, height * config->roi_bottom_ratio),
      cv::Point(width * 0.4, height * config->roi_top_ratio),
      cv::Point(width * 0.6, height * config->roi_top_ratio),
      cv::Point(width * config->roi_right_ratio, height * config->roi_bottom_ratio)};
  cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{pts}, cv::Scalar(255));
  return mask;
}

cv::Mat create_hsv_mask(const cv::Mat &frame, const LaneConfig *config) {
  cv::Mat hsv, mask_white, mask_yellow, combined;
  cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
  cv::inRange(hsv,
              cv::Scalar(config->white_hsv_lower[0], config->white_hsv_lower[1], config->white_hsv_lower[2]),
              cv::Scalar(config->white_hsv_upper[0], config->white_hsv_upper[1], config->white_hsv_upper[2]),
              mask_white);
  cv::inRange(hsv,
              cv::Scalar(config->yellow_hsv_lower[0], config->yellow_hsv_lower[1], config->yellow_hsv_lower[2]),
              cv::Scalar(config->yellow_hsv_upper[0], config->yellow_hsv_upper[1], config->yellow_hsv_upper[2]),
              mask_yellow);
  cv::bitwise_or(mask_white, mask_yellow, combined);
  return combined;
}

} // namespace

// ─────────────────────────────────────────────────────────────────
// Impl: 원본 main.cpp의 전역/루프 지역 상태들을 그대로 옮겨온 곳.
// (LaneConfig 4종, 현재 모드, 디바운스 카운터, 좌우 차선 temporal state)
// ─────────────────────────────────────────────────────────────────
struct ClassicalLaneDetector::Impl {
  LaneConfig cfg_sunny, cfg_overcast, cfg_rainy, cfg_night;
  LaneConfig cfg; // 현재 활성 config

  LightMode current_mode = MODE_SUNNY;
  LightMode pending_mode = MODE_SUNNY;
  int pending_mode_count = 0;

  LineSegment left, right;
  double conf_left = 0.0, conf_right = 0.0;
  int left_coast_count = 0, right_coast_count = 0;
  bool initialized = false;

  // ── 차선 폭 일관성 필터 상태 ──
  // 두 차선이 모두 안정적일 때 EMA로 학습한 기준 차선 폭(px, lookahead y 기준)
  double smoothed_lane_width = 0.0;
  bool lane_width_ready = false; // true가 되면 폭 일관성 필터 활성화

  // 디버그용: 가장 최근 프레임에서 쓰인 마스크들 (debug_mask_overlay()에서 사용)
  cv::Mat last_color_mask;
  cv::Mat last_roi_mask;

  std::string mode_label() const {
    switch (current_mode) {
    case MODE_SUNNY: return "MODE: SUNNY (Auto)";
    case MODE_OVERCAST: return "MODE: OVERCAST/FOG (Auto)";
    case MODE_RAINY: return "MODE: RAINY (Auto)";
    case MODE_NIGHT: return "MODE: NIGHT (Auto)";
    }
    return "MODE: ?";
  }
};

ClassicalLaneDetector::ClassicalLaneDetector(const std::string &sunny_path,
                                             const std::string &overcast_path,
                                             const std::string &rainy_path,
                                             const std::string &night_path)
    : impl_(std::make_unique<Impl>()) {
  if (!load_config(sunny_path.c_str(), &impl_->cfg_sunny))
    throw std::runtime_error("sunny config 로드 실패: " + sunny_path);
  if (!load_config(overcast_path.c_str(), &impl_->cfg_overcast))
    throw std::runtime_error("overcast config 로드 실패: " + overcast_path);
  if (!load_config(rainy_path.c_str(), &impl_->cfg_rainy))
    throw std::runtime_error("rainy config 로드 실패: " + rainy_path);
  if (!load_config(night_path.c_str(), &impl_->cfg_night))
    throw std::runtime_error("night config 로드 실패: " + night_path);
  impl_->cfg = impl_->cfg_sunny;
}

ClassicalLaneDetector::~ClassicalLaneDetector() = default;

std::string ClassicalLaneDetector::name() const { return "Classical CV (Auto Weather)"; }

LaneResult ClassicalLaneDetector::detect(const cv::Mat &frame) {
  Impl &s = *impl_;
  int width = frame.cols, height = frame.rows;

  // ── 복합 조도/채도 감지 (하늘 40% + 노면 60% 가중) ──
  cv::Rect sky_roi(0, 0, width, height / 2);
  cv::Rect road_roi(0, height / 2, width, height / 2);
  cv::Mat sky_gray, sky_hsv, road_gray, road_hsv;
  cv::cvtColor(frame(sky_roi), sky_gray, cv::COLOR_BGR2GRAY);
  cv::cvtColor(frame(sky_roi), sky_hsv, cv::COLOR_BGR2HSV);
  cv::cvtColor(frame(road_roi), road_gray, cv::COLOR_BGR2GRAY);
  cv::cvtColor(frame(road_roi), road_hsv, cv::COLOR_BGR2HSV);

  double sky_bright = cv::mean(sky_gray)[0];
  double road_bright = cv::mean(road_gray)[0];
  std::vector<cv::Mat> sky_ch, road_ch;
  cv::split(sky_hsv, sky_ch);
  cv::split(road_hsv, road_ch);
  double sky_sat = cv::mean(sky_ch[1])[0];
  double road_sat = cv::mean(road_ch[1])[0];

  double mean_brightness = 0.4 * sky_bright + 0.6 * road_bright;
  double mean_saturation = 0.4 * sky_sat + 0.6 * road_sat;

  // ── 1단계: target_mode 추정 ──
  LightMode target_mode = s.current_mode;
  if (mean_brightness < BRIGHTNESS_NIGHT_ENTER) {
    target_mode = MODE_NIGHT;
  } else if (mean_brightness < BRIGHTNESS_RAINY_ENTER) {
    target_mode = MODE_RAINY;
  } else if (mean_saturation <= SATURATION_OVERCAST_MAX) {
    target_mode = MODE_OVERCAST;
  } else if (mean_saturation >= SATURATION_SUNNY_MIN && mean_brightness >= BRIGHTNESS_SUNNY_ENTER) {
    target_mode = MODE_SUNNY;
  }

  // ── 2단계: N-Frame 디바운스 래치 ──
  if (target_mode == s.current_mode) {
    s.pending_mode_count = 0;
  } else {
    if (target_mode == s.pending_mode) {
      s.pending_mode_count++;
    } else {
      s.pending_mode = target_mode;
      s.pending_mode_count = 1;
    }
    if (s.pending_mode_count >= STABLE_FRAME_THRESHOLD) {
      s.current_mode = target_mode;
      s.pending_mode_count = 0;
      switch (s.current_mode) {
      case MODE_SUNNY: s.cfg = s.cfg_sunny; break;
      case MODE_OVERCAST: s.cfg = s.cfg_overcast; break;
      case MODE_RAINY: s.cfg = s.cfg_rainy; break;
      case MODE_NIGHT: s.cfg = s.cfg_night; break;
      }
    }
  }

  cv::Mat roi_mask = create_roi_mask(width, height, &s.cfg);
  cv::Mat color_mask = create_hsv_mask(frame, &s.cfg);

  int sky_height = static_cast<int>(height * s.cfg.roi_top_ratio);
  if (sky_height > 0 && sky_height < height)
    color_mask(cv::Rect(0, 0, width, sky_height)).setTo(0);

  // 디버그용: 이번 프레임에 실제로 쓰인 마스크 보관
  s.last_color_mask = color_mask.clone();
  s.last_roi_mask = roi_mask.clone();

  int ksize = s.cfg.gaussian_kernel_size;
  if (ksize % 2 == 0) ksize += 1;

  cv::Mat blurred_color, color_edges;
  cv::GaussianBlur(color_mask, blurred_color, cv::Size(ksize, ksize), 0);
  cv::Canny(blurred_color, color_edges, s.cfg.canny_low_threshold, s.cfg.canny_high_threshold);

  cv::Mat target_edges;
  if (s.current_mode == MODE_SUNNY) {
    target_edges = color_edges;
  } else {
    cv::Mat gray_frame, blurred_gray, gray_edges;
    cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray_frame, blurred_gray, cv::Size(ksize, ksize), 0);
    cv::Canny(blurred_gray, gray_edges, s.cfg.canny_low_threshold, s.cfg.canny_high_threshold);
    cv::bitwise_or(color_edges, gray_edges, target_edges);
  }

  cv::Mat roi_edges;
  cv::bitwise_and(target_edges, roi_mask, roi_edges);

  std::vector<cv::Vec4i> lines;
  cv::HoughLinesP(roi_edges, lines, s.cfg.hough_rho, s.cfg.hough_theta_deg * CV_PI / 180.0,
                   s.cfg.hough_threshold, s.cfg.hough_min_line_length, s.cfg.hough_max_line_gap);

  double left_m_sum = 0, left_b_sum = 0, left_len_sum = 0;
  double right_m_sum = 0, right_b_sum = 0, right_len_sum = 0;
  for (const auto &l : lines) {
    double x1 = l[0], y1 = l[1], x2 = l[2], y2 = l[3];
    if (x1 == x2) continue;
    double m = (y2 - y1) / (x2 - x1);
    if (fabs(m) < s.cfg.minimum_absolute_slope || fabs(m) > s.cfg.maximum_absolute_slope) continue;
    double b = y1 - m * x1;
    double len = sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
    double mid_y = (y1 + y2) / 2.0;
    double mid_x = (x1 + x2) / 2.0;

    if (m < 0 && x1 < width * 0.55 && x2 < width * 0.55) {
      // 이전 프레임에 추적 중인 좌측 차선이 있으면, 그 위치에서 너무 멀리
      // 떨어진 세그먼트(다른 차로/차량 윤곽선 등 노이즈)는 제외
      if (s.initialized && s.left.slope != 0.0) {
        double expected_x = calc_x_at_y(s.left.slope, s.left.intercept, mid_y);
        if (fabs(mid_x - expected_x) > s.cfg.max_track_deviation_px) continue;
      }
      left_m_sum += m * len; left_b_sum += b * len; left_len_sum += len;
    } else if (m > 0 && x1 > width * 0.45 && x2 > width * 0.45) {
      if (s.initialized && s.right.slope != 0.0) {
        double expected_x = calc_x_at_y(s.right.slope, s.right.intercept, mid_y);
        if (fabs(mid_x - expected_x) > s.cfg.max_track_deviation_px) continue;
      }
      right_m_sum += m * len; right_b_sum += b * len; right_len_sum += len;
    }
  }

  LineSegment curr_left{0, 0, left_len_sum, left_len_sum > 0};
  LineSegment curr_right{0, 0, right_len_sum, right_len_sum > 0};
  if (curr_left.valid) { curr_left.slope = left_m_sum / left_len_sum; curr_left.intercept = left_b_sum / left_len_sum; }
  if (curr_right.valid) { curr_right.slope = right_m_sum / right_len_sum; curr_right.intercept = right_b_sum / right_len_sum; }

  // ── 차선 폭 일관성 필터 ──
  const double lookahead_y = height * 0.65;
  const double tol = s.cfg.lane_width_tolerance_ratio;

  if (s.lane_width_ready) {
    const double w_min = s.smoothed_lane_width * (1.0 - tol);
    const double w_max = s.smoothed_lane_width * (1.0 + tol);

    // 1. 좌우 후보가 모두 있을 때 폭 검사
    if (curr_left.valid && curr_right.valid) {
      double lx_meas = calc_x_at_y(curr_left.slope, curr_left.intercept, lookahead_y);
      double rx_meas = calc_x_at_y(curr_right.slope, curr_right.intercept, lookahead_y);
      double measured_width = rx_meas - lx_meas;

      if (measured_width < w_min || measured_width > w_max) {
        // 폭이 기준을 벗어나면, 과거 궤적 대비 더 많이 튄 쪽을 범인으로 간주하고 기각
        double lx_expected = s.left.slope != 0.0 ? calc_x_at_y(s.left.slope, s.left.intercept, lookahead_y) : lx_meas;
        double rx_expected = s.right.slope != 0.0 ? calc_x_at_y(s.right.slope, s.right.intercept, lookahead_y) : rx_meas;
        
        double left_shift = fabs(lx_meas - lx_expected);
        double right_shift = fabs(rx_meas - rx_expected);

        if (right_shift > left_shift) {
          curr_right.valid = false;
        } else {
          curr_left.valid = false;
        }
      }
    }

    // 2. 한쪽만 있을 때, 반대쪽 과거 궤적을 이용해 검사
    if (curr_right.valid && !curr_left.valid && s.conf_left >= s.cfg.minimum_confidence && s.left.slope != 0.0) {
      double lx_expected = calc_x_at_y(s.left.slope, s.left.intercept, lookahead_y);
      double rx_meas = calc_x_at_y(curr_right.slope, curr_right.intercept, lookahead_y);
      double measured_width = rx_meas - lx_expected;
      if (measured_width < w_min || measured_width > w_max) {
        curr_right.valid = false;
      }
    }

    if (curr_left.valid && !curr_right.valid && s.conf_right >= s.cfg.minimum_confidence && s.right.slope != 0.0) {
      double lx_meas = calc_x_at_y(curr_left.slope, curr_left.intercept, lookahead_y);
      double rx_expected = calc_x_at_y(s.right.slope, s.right.intercept, lookahead_y);
      double measured_width = rx_expected - lx_meas;
      if (measured_width < w_min || measured_width > w_max) {
        curr_left.valid = false;
      }
    }
  }

  double alpha = s.cfg.temporal_alpha;

  if (!s.initialized) {
    s.left = curr_left;
    s.right = curr_right;
    s.left_coast_count = 0;
    s.right_coast_count = 0;
    s.initialized = true;
  } else {
    if (curr_left.valid) {
      s.left.slope = alpha * curr_left.slope + (1 - alpha) * s.left.slope;
      s.left.intercept = alpha * curr_left.intercept + (1 - alpha) * s.left.intercept;
      s.conf_left = std::min(1.0, s.conf_left + 0.2);
      s.left_coast_count = 0;
    } else if (s.left_coast_count < MAX_COAST_FRAMES) {
      s.left_coast_count++;
      s.conf_left = std::max(0.0, s.conf_left - 0.04);
    } else {
      s.conf_left = std::max(0.0, s.conf_left - 0.15);
    }

    if (curr_right.valid) {
      s.right.slope = alpha * curr_right.slope + (1 - alpha) * s.right.slope;
      s.right.intercept = alpha * curr_right.intercept + (1 - alpha) * s.right.intercept;
      s.conf_right = std::min(1.0, s.conf_right + 0.2);
      s.right_coast_count = 0;
    } else if (s.right_coast_count < MAX_COAST_FRAMES) {
      s.right_coast_count++;
      s.conf_right = std::max(0.0, s.conf_right - 0.04);
    } else {
      s.conf_right = std::max(0.0, s.conf_right - 0.15);
    }
  }

  // ── 기준 폭 학습 (두 차선 모두 안정적일 때만 EMA 업데이트) ──
  // 신뢰도가 높고 두 차선이 모두 유효한 프레임에서만 폭을 학습해
  // 노이즈가 심한 구간에서 기준값이 오염되는 것을 방지함
  const double confidence_threshold_for_learning = 0.7;
  if (s.conf_left >= confidence_threshold_for_learning &&
      s.conf_right >= confidence_threshold_for_learning &&
      s.left.slope != 0.0 && s.right.slope != 0.0) {
    double lx = calc_x_at_y(s.left.slope, s.left.intercept, lookahead_y);
    double rx = calc_x_at_y(s.right.slope, s.right.intercept, lookahead_y);
    double current_width = rx - lx;
    if (current_width > 50.0) { // 물리적으로 말이 안 되는 너무 좁은 폭은 학습 제외
      if (!s.lane_width_ready) {
        s.smoothed_lane_width = current_width; // 첫 번째 유효한 폭으로 초기화
        s.lane_width_ready = true;
        printf("[WidthFilter] ★ 기준 폭 초기화: %.1fpx (±%.0f%% = [%.1f, %.1f])\n",
               s.smoothed_lane_width, tol * 100,
               s.smoothed_lane_width * (1.0 - tol), s.smoothed_lane_width * (1.0 + tol));
      } else {
        // 분기/합류 차선이 서서히 벌어지는 경우 EMA가 오염되는 것을 방지하기 위해,
        // 현재 측정된 폭이 기준 폭의 ±10% 이내일 때만 기준 폭을 업데이트합니다.
        if (fabs(current_width - s.smoothed_lane_width) < s.smoothed_lane_width * 0.10) {
          s.smoothed_lane_width = 0.05 * current_width + 0.95 * s.smoothed_lane_width;
        } else {
          // 기준 폭과 너무 차이나는 폭은 오검출(분기차선 등)로 간주하고 학습 안함
        }
      }
    }
  }

  LaneResult result;
  result.left_valid = s.conf_left >= s.cfg.minimum_confidence;
  result.right_valid = s.conf_right >= s.cfg.minimum_confidence;
  result.left_slope = s.left.slope;
  result.left_intercept = s.left.intercept;
  result.right_slope = s.right.slope;
  result.right_intercept = s.right.intercept;
  result.conf_left = s.conf_left;
  result.conf_right = s.conf_right;
  result.status_label = s.mode_label();
  return result;
}

cv::Mat ClassicalLaneDetector::debug_mask_overlay() const {
  Impl &s = *impl_;
  if (s.last_color_mask.empty()) return cv::Mat();

  cv::Mat mask_bgr, roi_bgr, overlay;
  cv::cvtColor(s.last_color_mask, mask_bgr, cv::COLOR_GRAY2BGR);
  cv::cvtColor(s.last_roi_mask, roi_bgr, cv::COLOR_GRAY2BGR);
  cv::addWeighted(mask_bgr, 1.0, roi_bgr, 0.25, 0, overlay);

  int width = overlay.cols, height = overlay.rows;
  // 좌/우 판정 기준선 (코드 상 x < width*0.55 -> 좌측 후보, x > width*0.45 -> 우측 후보)
  cv::line(overlay, cv::Point(width * 0.45, 0), cv::Point(width * 0.45, height),
            cv::Scalar(0, 255, 0), 1);   // 우측 후보 시작선
  cv::line(overlay, cv::Point(width * 0.55, 0), cv::Point(width * 0.55, height),
            cv::Scalar(0, 140, 255), 1); // 좌측 후보 종료선

  cv::putText(overlay, "Color Mask (White+Yellow) + ROI + L/R split", cv::Point(10, 30),
              cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 128), 2);
  return overlay;
}
