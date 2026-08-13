#include "vehicle_config.h"
#include <cjson/cJSON.h>
#include <cstdio>
#include <cstdlib>

static double get_double_or(cJSON *json, const char *key, double fallback) {
  cJSON *item = cJSON_GetObjectItem(json, key);
  return (item && cJSON_IsNumber(item)) ? item->valuedouble : fallback;
}

bool load_vehicle_config(const std::string &filepath, VehicleConfig &config) {
  FILE *file = fopen(filepath.c_str(), "r");
  if (!file) {
    printf("[Error] vehicle config를 열 수 없습니다: %s\n", filepath.c_str());
    return false;
  }
  fseek(file, 0, SEEK_END);
  long length = ftell(file);
  fseek(file, 0, SEEK_SET);
  char *buffer = (char *)malloc(length + 1);
  size_t bytes_read = fread(buffer, 1, length, file);
  if (bytes_read != static_cast<size_t>(length)) {
    free(buffer);
    fclose(file);
    return false;
  }
  buffer[length] = '\0';
  fclose(file);

  cJSON *json = cJSON_Parse(buffer);
  free(buffer);
  if (!json) {
    printf("[Error] vehicle config JSON 파싱 실패: %s\n", filepath.c_str());
    return false;
  }

  // 값이 없으면 구조체 기본값을 그대로 사용 (필드 누락에 안전)
  config.lane_width_pixels = get_double_or(json, "lane_width_pixels", config.lane_width_pixels);
  config.lane_width_meters = get_double_or(json, "lane_width_meters", config.lane_width_meters);
  config.lookahead_y_ratio = get_double_or(json, "lookahead_y_ratio", config.lookahead_y_ratio);
  config.draw_y_bottom_ratio = get_double_or(json, "draw_y_bottom_ratio", config.draw_y_bottom_ratio);
  config.departure_warning_m = get_double_or(json, "departure_warning_m", config.departure_warning_m);
  config.departure_stop_m = get_double_or(json, "departure_stop_m", config.departure_stop_m);

  cJSON_Delete(json);
  return true;
}
