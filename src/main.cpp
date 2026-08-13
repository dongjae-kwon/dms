#include "classical_lane_detector.h"
#include "lane_detector.h"
#include "renderer.h"
#include "ufld_lane_detector.h"
#include "ufldv2_lane_detector.h"
#include "vehicle_config.h"
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>

// 사용 예 (경로를 직접 지정하고 싶을 때):
//   ./lane_app classical configs/lane_config_sunny.json configs/lane_config_overcast.json \
//                          configs/lane_config_rainy.json configs/lane_config_night.json
//   ./lane_app ufld models/ufld_culane_res18.onnx
//   ./lane_app ufldv2 models/ufldv2_culane_res18.onnx
//
// 매번 경로를 다 안 치고 싶으면 인자 없이 모드만 줘도 됨 (기본 경로 사용):
//   ./lane_app classical
//   ./lane_app ufld
//   ./lane_app ufldv2
static void print_usage(const char *prog) {
  printf("사용법:\n");
  printf("  %s classical [<sunny.json> <overcast.json> <rainy.json> <night.json>]\n", prog);
  printf("      (경로 생략 시 기본값: configs/lane_config_{sunny,overcast,rainy,night}.json)\n");
  printf("  %s ufld [<model.onnx>]\n", prog);
  printf("      (경로 생략 시 기본값: models/ufld_culane_res18.onnx)\n");
  printf("  %s ufldv2 [<model.onnx>]\n", prog);
  printf("      (경로 생략 시 기본값: models/ufldv2_culane_res18.onnx)\n");
}

int main(int argc, char **argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0); // Docker/파이프 환경에서 printf가 즉시 보이도록 라인버퍼링 강제

  if (argc < 2) {
    print_usage(argv[0]);
    return -1;
  }

  std::string mode = argv[1];
  std::unique_ptr<LaneDetector> detector;

  try {
    if (mode == "classical") {
      // 4개 경로를 다 줬으면 그걸 쓰고, 아니면(argc==2) 기본 경로로 대체
      std::string sunny = (argc >= 6) ? argv[2] : "configs/lane_config_sunny.json";
      std::string overcast = (argc >= 6) ? argv[3] : "configs/lane_config_overcast.json";
      std::string rainy = (argc >= 6) ? argv[4] : "configs/lane_config_rainy.json";
      std::string night = (argc >= 6) ? argv[5] : "configs/lane_config_night.json";
      if (argc != 2 && argc != 6) { print_usage(argv[0]); return -1; } // 일부만 준 경우는 에러
      detector = std::make_unique<ClassicalLaneDetector>(sunny, overcast, rainy, night);
    } else if (mode == "ufld") {
      std::string model_path = (argc >= 3) ? argv[2] : "models/ufld_culane_res18.onnx";
      detector = std::make_unique<UFLDLaneDetector>(model_path);
    } else if (mode == "ufldv2") {
      std::string model_path = (argc >= 3) ? argv[2] : "models/ufldv2_culane_res18.onnx";
      detector = std::make_unique<UFLDv2LaneDetector>(model_path);
    } else {
      print_usage(argv[0]);
      return -1;
    }
  } catch (const std::exception &e) {
    printf("[Error] 검출기 초기화 실패: %s\n", e.what());
    return -1;
  }

  printf("[Info] 검출기: %s\n", detector->name().c_str());

  VehicleConfig vcfg;
  if (!load_vehicle_config("configs/vehicle_config.json", vcfg)) {
    printf("[Warning] vehicle_config.json 로드 실패 - 기본값으로 진행합니다.\n");
  }

  cv::VideoCapture cap("good/sample_vid.mp4");
  if (!cap.isOpened()) {
    printf("[Error] 비디오 파일을 열 수 없습니다.\n");
    return -1;
  }

  int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  double fps = cap.get(cv::CAP_PROP_FPS);
  if (fps <= 0) fps = 10.0;

  const std::string output_path = "output_result.avi";
  cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                          fps, cv::Size(width, height));
  if (!writer.isOpened()) {
    printf("[Error] %s 를 쓸 수 없습니다 (VideoWriter 오픈 실패). "
           "코덱/FFmpeg 지원 여부를 확인하세요.\n", output_path.c_str());
    return -1;
  }

  // classical 모드일 때만: 원본 결과 + 색상마스크/ROI/좌우판정선을 나란히 보여주는
  // 디버그 영상도 같이 생성 (특정 방향 차선이 왜 안 잡히는지 확인용)
  const std::string debug_path = "debug_mask.avi";
  cv::VideoWriter debug_writer;
  if (mode == "classical") {
    debug_writer.open(debug_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps,
                       cv::Size(width * 2, height));
  }

  cv::Mat frame;
  int frame_idx = 0;
  while (cap.read(frame)) {
    if (frame.empty()) break;

    LaneResult result = detector->detect(frame);

    double offset_m = 0.0;
    cv::Mat display = render_frame(frame, result, vcfg, offset_m);

    // ── 디버그: 매 프레임 raw(필터 전) vs final(필터/coasting 후) 비교 출력 ──
    // (원인 파악 끝나면 주기를 다시 늘리거나 지워도 됨)
    printf("[Debug] frame=%d raw(L=%d,R=%d) final(L=%d,R=%d) conf_L=%.2f conf_R=%.2f status=%s\n",
           frame_idx, result.raw_left_valid, result.raw_right_valid,
           result.left_valid, result.right_valid, result.conf_left, result.conf_right,
           result.status_label.c_str());
    frame_idx++;

    writer.write(display);

    if (debug_writer.isOpened()) {
      if (auto *classical = dynamic_cast<ClassicalLaneDetector *>(detector.get())) {
        cv::Mat mask_overlay = classical->debug_mask_overlay();
        if (!mask_overlay.empty()) {
          cv::Mat side_by_side(height, width * 2, CV_8UC3);
          display.copyTo(side_by_side(cv::Rect(0, 0, width, height)));
          mask_overlay.copyTo(side_by_side(cv::Rect(width, 0, width, height)));
          debug_writer.write(side_by_side);
        }
      }
    }
  }

  cap.release();
  writer.release();
  debug_writer.release();

  std::filesystem::path abs_path = std::filesystem::absolute(output_path);
  printf("[Info] 처리 완료. 총 %d 프레임 처리됨.\n", frame_idx);
  printf("[Info] 결과 파일: %s\n", abs_path.string().c_str());
  if (mode == "classical") {
    std::filesystem::path abs_debug_path = std::filesystem::absolute(debug_path);
    printf("[Info] 마스크 디버그 파일: %s\n", abs_debug_path.string().c_str());
  }
  return 0;
}
