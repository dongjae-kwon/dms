#include "ufldv2_lane_detector.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace {
// ── Netron 그래프에서 역산한 CULane res18 표준 설정값 ──
constexpr int kInputW = 1600;
constexpr int kInputH = 800; // CurveLanes 모델(800x1600) 기준. CULane(320x1600)이면 320으로 되돌릴 것
constexpr int kNumCellRow = 200;
constexpr int kNumRow = 72;
constexpr int kNumCellCol = 100;
constexpr int kNumCol = 41;  // CurveLanes 기준 (CULane은 81)
constexpr int kNumLanes = 10; // CurveLanes 기준 (CULane은 4)
constexpr double kCropRatio = 0.6; // 원본 이미지 하단 60%만 잘라서 네트워크에 입력

constexpr int kLocRowSize = kNumCellRow * kNumRow * kNumLanes;   // 57600
constexpr int kExistRowSize = 2 * kNumRow * kNumLanes;           // 576
constexpr int kLocColSize = kNumCellCol * kNumCol * kNumLanes;   // 32400
constexpr int kExistColSize = 2 * kNumCol * kNumLanes;           // 648

constexpr float kMeanR = 0.485f, kMeanG = 0.456f, kMeanB = 0.406f;
constexpr float kStdR = 0.229f, kStdG = 0.224f, kStdB = 0.225f;

// row anchor: 원본 UFLDv2 CULane 설정 (0.42~1.0 구간, 이미지 높이 비율)
// col anchor: 0~1.0 구간, 이미지 너비 비율
// ⚠️ 공식 repo는 crop_ratio=0.6 크롭을 먼저 적용한 뒤 이 비율을 매기므로,
//    실제 세로 위치가 화면과 살짝 안 맞을 수 있습니다. debug_mask 영상으로
//    확인 후 필요하면 아래 두 lambda의 계산식을 조정하세요.
double row_anchor_frac(int i) { return 0.42 + i * (1.0 - 0.42) / (kNumRow - 1); }
double col_anchor_frac(int i) { return double(i) / (kNumCol - 1); }

bool fit_x_on_y(const std::vector<cv::Point2f> &pts, double &a, double &b) {
  if (pts.size() < 2) return false;
  double sum_y = 0, sum_x = 0, sum_yy = 0, sum_xy = 0;
  int n = static_cast<int>(pts.size());
  for (const auto &p : pts) {
    sum_y += p.y; sum_x += p.x;
    sum_yy += double(p.y) * p.y;
    sum_xy += double(p.x) * p.y;
  }
  double denom = n * sum_yy - sum_y * sum_y;
  if (fabs(denom) < 1e-6) return false;
  a = (n * sum_xy - sum_y * sum_x) / denom;
  b = (sum_x - a * sum_y) / n;
  return true;
}

double calc_x_at_y(double slope, double intercept, double y) {
  if (fabs(slope) < 1e-5) return 0.0;
  return (y - intercept) / slope;
}
} // namespace

UFLDv2LaneDetector::UFLDv2LaneDetector(const std::string &onnx_model_path,
                                       double temporal_alpha, double minimum_confidence,
                                       int max_coast_frames, double max_track_deviation_px)
    : env_(ORT_LOGGING_LEVEL_WARNING, "ufldv2_lane_detector"),
      session_(env_, onnx_model_path.c_str(), Ort::SessionOptions{nullptr}),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      temporal_alpha_(temporal_alpha), minimum_confidence_(minimum_confidence),
      max_coast_frames_(max_coast_frames), max_track_deviation_px_(max_track_deviation_px) {
  Ort::AllocatorWithDefaultOptions allocator;

  auto input_name_ptr = session_.GetInputNameAllocated(0, allocator);
  input_name_ = input_name_ptr.get();

  size_t out_count = session_.GetOutputCount();
  std::vector<std::string> all_output_names;

  for (size_t i = 0; i < out_count; ++i) {
    auto name_ptr = session_.GetOutputNameAllocated(i, allocator);
    all_output_names.push_back(name_ptr.get());

    // ── 진단용: 실제 shape을 콘솔에 출력 (인덱싱 공식이 맞는지 검증용) ──
    auto type_info = session_.GetOutputTypeInfo(i);
    auto shape = type_info.GetTensorTypeAndShapeInfo().GetShape();
    printf("[Debug] output[%zu] name=%s shape=[", i, all_output_names.back().c_str());
    for (size_t d = 0; d < shape.size(); ++d) printf("%s%lld", d ? "," : "", (long long)shape[d]);
    printf("]\n");
  }

  // 출력이 몇 개든 상관없이, 우리에게 필요한 4개(loc_row/loc_col/exist_row/
  // exist_col)만 이름으로 골라서 씀. 나머지(학습용 aux 출력 등)는 무시.
  const std::vector<std::string> required = {"loc_row", "loc_col", "exist_row", "exist_col"};
  for (const auto &req : required) {
    if (std::find(all_output_names.begin(), all_output_names.end(), req) == all_output_names.end())
      throw std::runtime_error("UFLDv2 모델에 필요한 출력 '" + req + "'가 없습니다. 위 [Debug] 로그의 "
                                "이름 목록을 확인하세요.");
    output_names_.push_back(req);
  }
  // 이름/순서에 의존하지 않고, 그래프에서 역산한 텐서 크기로 역할을 매칭
  // (loc_row/exist_row/loc_col/exist_col 각각의 element 개수로 구분) — 순서가
  // 바뀌어도 안전하게 동작합니다.
}

UFLDv2LaneDetector::~UFLDv2LaneDetector() = default;

std::vector<float> UFLDv2LaneDetector::preprocess(const cv::Mat &frame) {
  // 원본 이미지 하단 60%만 잘라서 네트워크에 입력 (원본 저장소의 crop_ratio 전처리)
  int crop_top = static_cast<int>(frame.rows * (1.0 - kCropRatio));
  cv::Rect crop_rect(0, crop_top, frame.cols, frame.rows - crop_top);
  cv::Mat cropped = frame(crop_rect);

  cv::Mat resized, rgb, float_img;
  cv::resize(cropped, resized, cv::Size(kInputW, kInputH));
  cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
  rgb.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

  std::vector<float> chw(3 * kInputH * kInputW);
  std::vector<cv::Mat> channels(3);
  cv::split(float_img, channels);

  auto normalize_channel = [](cv::Mat &ch, float mean, float std_) { ch = (ch - mean) / std_; };
  normalize_channel(channels[0], kMeanR, kStdR);
  normalize_channel(channels[1], kMeanG, kStdG);
  normalize_channel(channels[2], kMeanB, kStdB);

  for (int c = 0; c < 3; ++c)
    memcpy(chw.data() + c * kInputH * kInputW, channels[c].data, kInputH * kInputW * sizeof(float));
  return chw;
}

UFLDv2LaneDetector::RawDetection UFLDv2LaneDetector::postprocess(
    const float *loc_row, const float *exist_row, const float *loc_col,
    const float *exist_col, int frame_w, int frame_h) {

  // 인덱싱 헬퍼: [C, N, L] 레이아웃 (channel-major, PyTorch 기본 export 순서)
  auto loc_row_at = [&](int cell, int row, int lane) {
    return loc_row[(cell * kNumRow + row) * kNumLanes + lane];
  };
  auto exist_row_at = [&](int cls, int row, int lane) {
    return exist_row[(cls * kNumRow + row) * kNumLanes + lane];
  };
  auto loc_col_at = [&](int cell, int col, int lane) {
    return loc_col[(cell * kNumCol + col) * kNumLanes + lane];
  };
  auto exist_col_at = [&](int cls, int col, int lane) {
    return exist_col[(cls * kNumCol + col) * kNumLanes + lane];
  };

  std::vector<std::vector<cv::Point2f>> lane_points(kNumLanes);
  std::vector<double> lane_x_sum(kNumLanes, 0.0);
  std::vector<int> lane_x_count(kNumLanes, 0);

  const bool dump = (debug_dump_frames_left_ > 0);
  if (dump) printf("[Debug] ── postprocess dump (frame_w=%d frame_h=%d) ──\n", frame_w, frame_h);

  for (int lane = 0; lane < kNumLanes; ++lane) {
    // ── row anchor 기반 포인트 (가로로 누운 구간, 대부분의 직선 도로) ──
    for (int r = 0; r < kNumRow; ++r) {
      float e0 = exist_row_at(0, r, lane), e1 = exist_row_at(1, r, lane);
      if (e1 <= e0) continue; // 존재 안 함

      double max_logit = -1e18, sum_exp = 0;
      for (int c = 0; c < kNumCellRow; ++c) max_logit = std::max(max_logit, (double)loc_row_at(c, r, lane));
      std::vector<double> probs(kNumCellRow);
      for (int c = 0; c < kNumCellRow; ++c) { probs[c] = std::exp(loc_row_at(c, r, lane) - max_logit); sum_exp += probs[c]; }
      double expectation = 0;
      for (int c = 0; c < kNumCellRow; ++c) expectation += (probs[c] / sum_exp) * c;

      double x_norm = expectation / (kNumCellRow - 1);
      double x_px = x_norm * frame_w;
      // row_anchor_frac()은 crop된 영역(하단 60%) 내에서의 비율이므로 원본 프레임 좌표로 역변환
      double crop_top = frame_h * (1.0 - kCropRatio);
      double crop_h = frame_h * kCropRatio;
      double y_px = crop_top + row_anchor_frac(r) * crop_h;

      if (dump && lane_x_count[lane] < 5) {
        printf("[Debug]   lane=%d row=%d e0=%.2f e1=%.2f expectation=%.1f/%d x_px=%.1f y_px=%.1f\n",
               lane, r, e0, e1, expectation, kNumCellRow - 1, x_px, y_px);
      }

      lane_points[lane].emplace_back(float(x_px), float(y_px));
      lane_x_sum[lane] += x_px;
      lane_x_count[lane]++;
    }
    // ── col anchor 기반 포인트 (급커브 등 세로로 누운 구간 보강) ──
    for (int c = 0; c < kNumCol; ++c) {
      if (exist_col_at(1, c, lane) <= exist_col_at(0, c, lane)) continue;

      double max_logit = -1e18, sum_exp = 0;
      for (int cell = 0; cell < kNumCellCol; ++cell) max_logit = std::max(max_logit, (double)loc_col_at(cell, c, lane));
      std::vector<double> probs(kNumCellCol);
      for (int cell = 0; cell < kNumCellCol; ++cell) { probs[cell] = std::exp(loc_col_at(cell, c, lane) - max_logit); sum_exp += probs[cell]; }
      double expectation = 0;
      for (int cell = 0; cell < kNumCellCol; ++cell) expectation += (probs[cell] / sum_exp) * cell;

      double y_norm = expectation / (kNumCellCol - 1);
      double crop_top = frame_h * (1.0 - kCropRatio);
      double crop_h = frame_h * kCropRatio;
      double y_px = crop_top + y_norm * crop_h;
      double x_px = col_anchor_frac(c) * frame_w;

      lane_points[lane].emplace_back(float(x_px), float(y_px));
      lane_x_sum[lane] += x_px;
      lane_x_count[lane]++;
    }
  }

  if (dump) {
    for (int lane = 0; lane < kNumLanes; ++lane)
      printf("[Debug]   lane=%d total_points=%d avg_x=%.1f\n", lane, lane_x_count[lane],
             lane_x_count[lane] ? lane_x_sum[lane] / lane_x_count[lane] : -1.0);
  }

  double center_x = frame_w / 2.0;
  int best_left = -1, best_right = -1;
  double best_left_dist = 1e18, best_right_dist = 1e18;
  for (int lane = 0; lane < kNumLanes; ++lane) {
    if (lane_x_count[lane] < 2) continue;
    double avg_x = lane_x_sum[lane] / lane_x_count[lane];
    if (avg_x < center_x) {
      double d = center_x - avg_x;
      if (d < best_left_dist) { best_left_dist = d; best_left = lane; }
    } else {
      double d = avg_x - center_x;
      if (d < best_right_dist) { best_right_dist = d; best_right = lane; }
    }
  }

  RawDetection raw;
  if (best_left >= 0) {
    double a, b;
    if (fit_x_on_y(lane_points[best_left], a, b) && fabs(a) > 1e-9) {
      raw.left_valid = true; raw.left_slope = 1.0 / a; raw.left_intercept = -b / a;
    }
  }
  if (best_right >= 0) {
    double a, b;
    if (fit_x_on_y(lane_points[best_right], a, b) && fabs(a) > 1e-9) {
      raw.right_valid = true; raw.right_slope = 1.0 / a; raw.right_intercept = -b / a;
    }
  }
  return raw;
}

LaneResult UFLDv2LaneDetector::detect(const cv::Mat &frame) {
  std::vector<float> input_tensor_values = preprocess(frame);
  std::array<int64_t, 4> input_shape{1, 3, kInputH, kInputW};

  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      memory_info_, input_tensor_values.data(), input_tensor_values.size(),
      input_shape.data(), input_shape.size());

  const char *input_names[] = {input_name_.c_str()};
  std::vector<const char *> output_name_cstrs;
  for (auto &n : output_names_) output_name_cstrs.push_back(n.c_str());

  auto output_tensors = session_.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1,
                                      output_name_cstrs.data(), output_name_cstrs.size());

  // 텐서 크기로 역할 매칭 (이름/순서 의존 X)
  const float *loc_row = nullptr, *exist_row = nullptr, *loc_col = nullptr, *exist_col = nullptr;
  for (auto &t : output_tensors) {
    auto shape = t.GetTensorTypeAndShapeInfo().GetShape();
    int64_t total = 1;
    for (auto d : shape) if (d > 0) total *= d;
    const float *data = t.GetTensorData<float>();
    if (total == kLocRowSize) loc_row = data;
    else if (total == kExistRowSize) exist_row = data;
    else if (total == kLocColSize) loc_col = data;
    else if (total == kExistColSize) exist_col = data;
  }
  if (!loc_row || !exist_row || !loc_col || !exist_col)
    throw std::runtime_error("UFLDv2 출력 텐서 크기가 예상(57600/576/32400/648)과 다릅니다. "
                              "모델이 res18 CULane 320x1600이 맞는지 확인하세요.");

  RawDetection raw = postprocess(loc_row, exist_row, loc_col, exist_col, frame.cols, frame.rows);
  if (debug_dump_frames_left_ > 0) debug_dump_frames_left_--;
  const bool raw_left_before_filter = raw.left_valid;
  const bool raw_right_before_filter = raw.right_valid;

  // ── 교차/이상치 방지 ──
  // 실제로 화면에 그려지는 범위(대략 lookahead ~ 하단)의 양 끝점에서 모두
  // 검사. 검출된 포인트들이 좁은 y구간에만 몰려있으면, 화면 끝까지
  // 연장(extrapolate)했을 때 한쪽 지점에서만 봐선 놓칠 수 있는 교차가 생김.
  double ref_y_near = frame.rows * 0.65; // 대략적인 lookahead 지점
  double ref_y_far = frame.rows * 0.95;  // 대략적인 하단 지점
  double ref_y = ref_y_far; // 이하 트랙 이탈(이상치) 검사에는 하단 기준점 사용

  auto crosses = [&](const RawDetection &d) {
    double lx1 = calc_x_at_y(d.left_slope, d.left_intercept, ref_y_near);
    double rx1 = calc_x_at_y(d.right_slope, d.right_intercept, ref_y_near);
    double lx2 = calc_x_at_y(d.left_slope, d.left_intercept, ref_y_far);
    double rx2 = calc_x_at_y(d.right_slope, d.right_intercept, ref_y_far);
    return lx1 >= rx1 || lx2 >= rx2;
  };

  if (raw.left_valid && raw.right_valid && crosses(raw)) {
    printf("[Debug] CROSS DETECTED -> 이번 프레임 raw 검출 폐기 (L slope=%.4f b=%.1f / R slope=%.4f b=%.1f)\n",
           raw.left_slope, raw.left_intercept, raw.right_slope, raw.right_intercept);
    raw.left_valid = false; raw.right_valid = false;
  }
  if (conf_left_ > 0.0 && raw.left_valid) {
    double expected = calc_x_at_y(smoothed_left_slope_, smoothed_left_intercept_, ref_y);
    double actual = calc_x_at_y(raw.left_slope, raw.left_intercept, ref_y);
    if (fabs(actual - expected) > max_track_deviation_px_) raw.left_valid = false;
  }
  if (conf_right_ > 0.0 && raw.right_valid) {
    double expected = calc_x_at_y(smoothed_right_slope_, smoothed_right_intercept_, ref_y);
    double actual = calc_x_at_y(raw.right_slope, raw.right_intercept, ref_y);
    if (fabs(actual - expected) > max_track_deviation_px_) raw.right_valid = false;
  }

  if (!initialized_) {
    smoothed_left_slope_ = raw.left_slope; smoothed_left_intercept_ = raw.left_intercept;
    smoothed_right_slope_ = raw.right_slope; smoothed_right_intercept_ = raw.right_intercept;
    initialized_ = true;
  } else {
    if (raw.left_valid) {
      if (conf_left_ <= 0.0) { // 신뢰도가 바닥난 상태에서 새로 찾았을 때 궤도 초기화 (깜빡임/Deadlock 방지)
        smoothed_left_slope_ = raw.left_slope;
        smoothed_left_intercept_ = raw.left_intercept;
      } else {
        smoothed_left_slope_ = temporal_alpha_ * raw.left_slope + (1 - temporal_alpha_) * smoothed_left_slope_;
        smoothed_left_intercept_ = temporal_alpha_ * raw.left_intercept + (1 - temporal_alpha_) * smoothed_left_intercept_;
      }
      conf_left_ = std::min(1.0, conf_left_ + 0.2);
      left_coast_count_ = 0;
    } else if (left_coast_count_ < max_coast_frames_) {
      left_coast_count_++; conf_left_ = std::max(0.0, conf_left_ - 0.04);
    } else {
      conf_left_ = std::max(0.0, conf_left_ - 0.15);
    }

    if (raw.right_valid) {
      if (conf_right_ <= 0.0) { // 신뢰도가 바닥난 상태에서 새로 찾았을 때 궤도 초기화 (깜빡임/Deadlock 방지)
        smoothed_right_slope_ = raw.right_slope;
        smoothed_right_intercept_ = raw.right_intercept;
      } else {
        smoothed_right_slope_ = temporal_alpha_ * raw.right_slope + (1 - temporal_alpha_) * smoothed_right_slope_;
        smoothed_right_intercept_ = temporal_alpha_ * raw.right_intercept + (1 - temporal_alpha_) * smoothed_right_intercept_;
      }
      conf_right_ = std::min(1.0, conf_right_ + 0.2);
      right_coast_count_ = 0;
    } else if (right_coast_count_ < max_coast_frames_) {
      right_coast_count_++; conf_right_ = std::max(0.0, conf_right_ - 0.04);
    } else {
      conf_right_ = std::max(0.0, conf_right_ - 0.15);
    }
  }

  LaneResult result;
  result.left_valid = conf_left_ >= minimum_confidence_;
  result.right_valid = conf_right_ >= minimum_confidence_;
  result.left_slope = smoothed_left_slope_;
  result.left_intercept = smoothed_left_intercept_;
  result.right_slope = smoothed_right_slope_;
  result.right_intercept = smoothed_right_intercept_;
  result.conf_left = conf_left_;
  result.conf_right = conf_right_;
  result.raw_left_valid = raw_left_before_filter;
  result.raw_right_valid = raw_right_before_filter;
  result.status_label = "UFLDv2 (CurveLanes res18)";
  return result;
}
