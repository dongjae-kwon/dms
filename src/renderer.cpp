#include "renderer.h"
#include <cmath>
#include <cstdio>

namespace {
double calc_x_at_y(double slope, double intercept, double y) {
  if (fabs(slope) < 1e-5) return 0.0;
  return (y - intercept) / slope;
}
} // namespace

cv::Mat render_frame(const cv::Mat &frame, const LaneResult &result,
                      const VehicleConfig &vcfg, double &out_offset_m) {
  int width = frame.cols, height = frame.rows;
  double y_bottom = height * vcfg.draw_y_bottom_ratio;
  double y_lookahead = height * vcfg.lookahead_y_ratio;

  double left_x_look = calc_x_at_y(result.left_slope, result.left_intercept, y_lookahead);
  double right_x_look = calc_x_at_y(result.right_slope, result.right_intercept, y_lookahead);

  double img_center_x = width / 2.0;
  double lane_center_x = img_center_x;

  if (result.left_valid && result.right_valid) {
    lane_center_x = (left_x_look + right_x_look) / 2.0;
  } else if (result.left_valid) {
    lane_center_x = left_x_look + (vcfg.lane_width_pixels / 2.0);
  } else if (result.right_valid) {
    lane_center_x = right_x_look - (vcfg.lane_width_pixels / 2.0);
  }

  double offset_pixel = img_center_x - lane_center_x;
  double meter_per_pixel = vcfg.lane_width_meters / vcfg.lane_width_pixels;
  out_offset_m = offset_pixel * meter_per_pixel;

  cv::Mat display = frame.clone();

  if (result.left_valid) {
    double x1 = calc_x_at_y(result.left_slope, result.left_intercept, y_bottom);
    cv::line(display, cv::Point(x1, y_bottom), cv::Point(left_x_look, y_lookahead),
              cv::Scalar(255, 0, 0), 3);
  }
  if (result.right_valid) {
    double x1 = calc_x_at_y(result.right_slope, result.right_intercept, y_bottom);
    cv::line(display, cv::Point(x1, y_bottom), cv::Point(right_x_look, y_lookahead),
              cv::Scalar(0, 0, 255), 3);
  }
  cv::line(display, cv::Point(0, y_lookahead), cv::Point(width, y_lookahead),
            cv::Scalar(0, 255, 255), 1);

  char text_buf[160];
  snprintf(text_buf, sizeof(text_buf), "Offset: %.2fm | Conf (L:%.1f, R:%.1f)",
           out_offset_m, result.conf_left, result.conf_right);
  cv::putText(display, text_buf, cv::Point(30, 40), cv::FONT_HERSHEY_SIMPLEX, 0.7,
              cv::Scalar(0, 255, 0), 2);

  cv::putText(display, result.status_label, cv::Point(30, 70), cv::FONT_HERSHEY_SIMPLEX,
              0.65, cv::Scalar(0, 220, 255), 2);

  if (!result.left_valid && !result.right_valid) {
    cv::putText(display, "[WARNING] LANE TRACKING LOST!", cv::Point(30, 90),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 165, 255), 2);
  } else if (fabs(out_offset_m) > vcfg.departure_stop_m) {
    cv::putText(display, "[WARNING] STOP / CRITICAL DEPARTURE!", cv::Point(30, 90),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
  } else if (fabs(out_offset_m) > vcfg.departure_warning_m) {
    cv::putText(display, "[WARNING] LANE DEPARTURE!", cv::Point(30, 90),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 165, 255), 2);
  }

  return display;
}
