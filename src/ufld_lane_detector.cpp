#include "ufld_lane_detector.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace {
// ── 표준 UFLD + CULane 설정값 (모델이 다르면 반드시 이 값들을 교체) ──
constexpr int kInputW = 800;
constexpr int kInputH = 288;
constexpr int kGridingNum = 200;     // 가로 방향 grid cell 개수
constexpr int kClsNumPerLane = 18;   // lane 하나당 row anchor 개수
constexpr int kNumLanes = 4;         // 모델이 예측하는 전체 lane 슬롯 수 (좌2+우2)

// CULane row anchor (원본 config, 입력 height=288 기준 y좌표)
const std::vector<int> kRowAnchor = {121, 131, 141, 150, 160, 170, 180, 189,
                                      199, 209, 219, 229, 239, 249, 259, 268,
                                      278, 287};

constexpr float kMeanR = 0.485f, kMeanG = 0.456f, kMeanB = 0.406f;
constexpr float kStdR = 0.229f, kStdG = 0.224f, kStdB = 0.225f;

// x = a*y + b 최소자승 피팅. 점이 2개 미만이면 valid=false.
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

UFLDLaneDetector::UFLDLaneDetector(const std::string &onnx_model_path,
                                   double temporal_alpha, double minimum_confidence,
                                   int max_coast_frames, double max_track_deviation_px)
    : env_(ORT_LOGGING_LEVEL_WARNING, "ufld_lane_detector"),
      session_(env_, onnx_model_path.c_str(), Ort::SessionOptions{nullptr}),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      temporal_alpha_(temporal_alpha), minimum_confidence_(minimum_confidence),
      max_coast_frames_(max_coast_frames), max_track_deviation_px_(max_track_deviation_px) {
  Ort::AllocatorWithDefaultOptions allocator;

  // onnxruntime 버전에 따라 GetInputNameAllocated API 유무가 다를 수 있음.
  // 최신 버전(1.13+) 기준 예시:
  auto input_name_ptr = session_.GetInputNameAllocated(0, allocator);
  auto output_name_ptr = session_.GetOutputNameAllocated(0, allocator);
  input_name_ = input_name_ptr.get();
  output_name_ = output_name_ptr.get();
}

UFLDLaneDetector::~UFLDLaneDetector() = default;

std::vector<float> UFLDLaneDetector::preprocess(const cv::Mat &frame) {
  cv::Mat resized, rgb, float_img;
  cv::resize(frame, resized, cv::Size(kInputW, kInputH));
  cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
  rgb.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

  std::vector<float> chw(3 * kInputH * kInputW);
  std::vector<cv::Mat> channels(3);
  cv::split(float_img, channels); // R,G,B 순서

  auto normalize_channel = [](cv::Mat &ch, float mean, float std_) {
    ch = (ch - mean) / std_;
  };
  normalize_channel(channels[0], kMeanR, kStdR);
  normalize_channel(channels[1], kMeanG, kStdG);
  normalize_channel(channels[2], kMeanB, kStdB);

  for (int c = 0; c < 3; ++c)
    memcpy(chw.data() + c * kInputH * kInputW, channels[c].data,
           kInputH * kInputW * sizeof(float));
  return chw;
}

UFLDLaneDetector::RawDetection UFLDLaneDetector::postprocess(const std::vector<float> &output,
                                                              int frame_w, int frame_h) {
  // output shape: [1, kGridingNum + 1, kClsNumPerLane, kNumLanes]
  // (마지막 클래스 인덱스 kGridingNum은 "차선 없음" 배경 클래스)
  const int num_cls = kGridingNum + 1;
  auto at = [&](int cls, int row, int lane) -> float {
    return output[(cls * kClsNumPerLane + row) * kNumLanes + lane];
  };

  // col_sample: 원본 UFLD demo와 동일하게 0~(kInputW-1) 구간을 균등 분할
  std::vector<double> col_sample(kGridingNum);
  for (int i = 0; i < kGridingNum; ++i)
    col_sample[i] = i * (kInputW - 1) / double(kGridingNum - 1);

  // 각 lane 슬롯별로 (x,y) 포인트 목록과 평균 x 위치를 모음
  std::vector<std::vector<cv::Point2f>> lane_points(kNumLanes);
  std::vector<double> lane_avg_x(kNumLanes, -1.0);
  std::vector<int> lane_valid_count(kNumLanes, 0);

  for (int lane = 0; lane < kNumLanes; ++lane) {
    double x_sum = 0;
    int count = 0;
    for (int row = 0; row < kClsNumPerLane; ++row) {
      // softmax + argmax로 배경 여부 판단
      std::vector<float> logits(num_cls);
      for (int c = 0; c < num_cls; ++c) logits[c] = at(c, row, lane);
      int argmax = int(std::max_element(logits.begin(), logits.end()) - logits.begin());
      if (argmax == kGridingNum) continue; // 배경(차선 없음)

      // griding_num개 클래스(배경 제외)에 대해 softmax 기대값(expectation)으로
      // 서브픽셀 위치 추정 (원본 UFLD 후처리와 동일한 방식)
      double max_logit = *std::max_element(logits.begin(), logits.begin() + kGridingNum);
      double sum_exp = 0;
      std::vector<double> probs(kGridingNum);
      for (int c = 0; c < kGridingNum; ++c) {
        probs[c] = std::exp(logits[c] - max_logit);
        sum_exp += probs[c];
      }
      double loc = 0;
      for (int c = 0; c < kGridingNum; ++c) loc += (probs[c] / sum_exp) * c;

      double x_800 = col_sample[std::min(kGridingNum - 1, std::max(0, (int)std::round(loc)))];
      double x_px = x_800 * frame_w / double(kInputW);
      double y_px = kRowAnchor[row] * frame_h / double(kInputH);

      lane_points[lane].emplace_back(float(x_px), float(y_px));
      x_sum += x_px;
      count++;
    }
    lane_valid_count[lane] = count;
    if (count > 0) lane_avg_x[lane] = x_sum / count;
  }

  // 화면 중심에 가장 가까운 좌/우 lane 슬롯을 ego-lane으로 선택
  double center_x = frame_w / 2.0;
  int best_left = -1, best_right = -1;
  double best_left_dist = 1e18, best_right_dist = 1e18;
  for (int lane = 0; lane < kNumLanes; ++lane) {
    if (lane_valid_count[lane] < 2) continue;
    double avg_x = lane_avg_x[lane];
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
      raw.left_valid = true;
      raw.left_slope = 1.0 / a;
      raw.left_intercept = -b / a;
    }
  }
  if (best_right >= 0) {
    double a, b;
    if (fit_x_on_y(lane_points[best_right], a, b) && fabs(a) > 1e-9) {
      raw.right_valid = true;
      raw.right_slope = 1.0 / a;
      raw.right_intercept = -b / a;
    }
  }
  return raw;
}

LaneResult UFLDLaneDetector::detect(const cv::Mat &frame) {
  std::vector<float> input_tensor_values = preprocess(frame);
  std::array<int64_t, 4> input_shape{1, 3, kInputH, kInputW};

  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      memory_info_, input_tensor_values.data(), input_tensor_values.size(),
      input_shape.data(), input_shape.size());

  const char *input_names[] = {input_name_.c_str()};
  const char *output_names[] = {output_name_.c_str()};

  auto output_tensors = session_.Run(Ort::RunOptions{nullptr}, input_names,
                                      &input_tensor, 1, output_names, 1);

  float *out_data = output_tensors.front().GetTensorMutableData<float>();
  size_t out_size = (kGridingNum + 1) * kClsNumPerLane * kNumLanes;
  std::vector<float> output(out_data, out_data + out_size);

  RawDetection raw = postprocess(output, frame.cols, frame.rows);
  const bool raw_left_before_filter = raw.left_valid;   // 진단용: 필터링 전 원본
  const bool raw_right_before_filter = raw.right_valid; // 진단용: 필터링 전 원본

  // ── 교차/이상치 방지 ──
  // 실제로 화면에 그려지는 범위(대략 lookahead ~ 하단)의 양 끝점에서 모두
  // 검사. 한 지점만 보면 그 지점에서는 안 겹쳐도 다른 구간에서 넘어가는
  // 경우를 놓칠 수 있음.
  double ref_y_near = frame.rows * 0.65;
  double ref_y_far = frame.rows * 0.95;
  double ref_y = ref_y_far; // 이하 트랙 이탈(이상치) 검사에는 하단 기준점 사용

  auto crosses = [&](const RawDetection &d) {
    double lx1 = calc_x_at_y(d.left_slope, d.left_intercept, ref_y_near);
    double rx1 = calc_x_at_y(d.right_slope, d.right_intercept, ref_y_near);
    double lx2 = calc_x_at_y(d.left_slope, d.left_intercept, ref_y_far);
    double rx2 = calc_x_at_y(d.right_slope, d.right_intercept, ref_y_far);
    return lx1 >= rx1 || lx2 >= rx2;
  };

  if (raw.left_valid && raw.right_valid && crosses(raw)) {
    raw.left_valid = false;
    raw.right_valid = false;
  }

  // (2) 트랙에 이미 신뢰도가 쌓여있으면(conf > 0), 그 위치에서 너무 멀리 튄
  //     이번 프레임 검출은 버림. conf가 0인 상태(부트스트랩 직후 or 트랙을
  //     완전히 잃은 직후)에는 게이팅을 끄고 새로 자유롭게 재획득하게 둠 —
  //     안 그러면 smoothed 값이 0으로 고정된 채 영원히 거부만 하는 deadlock에 빠짐.
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

  // ── temporal smoothing (classical_lane_detector.cpp와 동일한 EMA + coasting 방식) ──
  if (!initialized_) {
    smoothed_left_slope_ = raw.left_slope; smoothed_left_intercept_ = raw.left_intercept;
    smoothed_right_slope_ = raw.right_slope; smoothed_right_intercept_ = raw.right_intercept;
    initialized_ = true;
  } else {
    if (raw.left_valid) {
      smoothed_left_slope_ = temporal_alpha_ * raw.left_slope + (1 - temporal_alpha_) * smoothed_left_slope_;
      smoothed_left_intercept_ = temporal_alpha_ * raw.left_intercept + (1 - temporal_alpha_) * smoothed_left_intercept_;
      conf_left_ = std::min(1.0, conf_left_ + 0.2);
      left_coast_count_ = 0;
    } else if (left_coast_count_ < max_coast_frames_) {
      left_coast_count_++;
      conf_left_ = std::max(0.0, conf_left_ - 0.04);
    } else {
      conf_left_ = std::max(0.0, conf_left_ - 0.15);
    }

    if (raw.right_valid) {
      smoothed_right_slope_ = temporal_alpha_ * raw.right_slope + (1 - temporal_alpha_) * smoothed_right_slope_;
      smoothed_right_intercept_ = temporal_alpha_ * raw.right_intercept + (1 - temporal_alpha_) * smoothed_right_intercept_;
      conf_right_ = std::min(1.0, conf_right_ + 0.2);
      right_coast_count_ = 0;
    } else if (right_coast_count_ < max_coast_frames_) {
      right_coast_count_++;
      conf_right_ = std::max(0.0, conf_right_ - 0.04);
    } else {
      conf_right_ = std::max(0.0, conf_right_ - 0.15);
    }
  }

  LaneResult result;
  // 히스테리시스: 이미 화면에 떠있던 선은 원래 기준(minimum_confidence_)의
  // 절반까지 떨어져야 사라지고, 새로 뜨려는 선은 원래 기준을 그대로 적용.
  // conf가 임계값 근처에서 오르내릴 때 매 프레임 뜨고 사라지는 깜빡임을 줄임.
  double hide_threshold = minimum_confidence_ * 0.5;
  result.left_valid = was_left_visible_ ? (conf_left_ >= hide_threshold) : (conf_left_ >= minimum_confidence_);
  result.right_valid = was_right_visible_ ? (conf_right_ >= hide_threshold) : (conf_right_ >= minimum_confidence_);
  was_left_visible_ = result.left_valid;
  was_right_visible_ = result.right_valid;
  result.left_slope = smoothed_left_slope_;
  result.left_intercept = smoothed_left_intercept_;
  result.right_slope = smoothed_right_slope_;
  result.right_intercept = smoothed_right_intercept_;
  result.conf_left = conf_left_;
  result.conf_right = conf_right_;
  result.raw_left_valid = raw_left_before_filter;
  result.raw_right_valid = raw_right_before_filter;
  result.status_label = "UFLD (CULane)";
  return result;
}
