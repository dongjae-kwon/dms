cmake_minimum_required(VERSION 3.16)
project(dms_project VERSION 1.0.0 LANGUAGES CXX C)

# 1. C++ 표준 설정
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 2. 필수 패키지 검색 (OpenCV & Threads & cJSON)
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_library(CJSON_LIBRARY
    NAMES cjson
    REQUIRED
)

# ── onnxruntime ──
# vcpkg를 쓴다면 find_package(onnxruntime CONFIG REQUIRED)로 대체 가능.
# 여기서는 공식 tarball을 압축 해제해서 쓰는 걸 기준으로 함.
set(ONNXRUNTIME_ROOT "/opt/onnxruntime" CACHE PATH "onnxruntime 설치 경로")
find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    PATHS ${ONNXRUNTIME_ROOT}/lib
    REQUIRED
)

# 3. 헤더 파일 경로 포함
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_ROOT}/include
)

# 4. 소스 파일 수집 (src 디렉터리 내부의 모든 .cpp 파일)
file(GLOB_RECURSE SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

if(NOT SOURCES)
    message(WARNING "src 디렉터리에 .cpp 소스 파일이 없습니다.")
endif()

# 5. 실행 바이너리(dms_node) 생성
add_executable(dms_node ${SOURCES})

# 6. 라이브러리 링크
target_link_libraries(dms_node
    PRIVATE
        ${OpenCV_LIBS}
        Threads::Threads
        ${CJSON_LIBRARY}
        ${ONNXRUNTIME_LIBRARY}
)

# 7. 빌드 정보 출력
message(STATUS "----------------------------------------------------")
message(STATUS "Project Name       : ${PROJECT_NAME}")
message(STATUS "Build Type         : ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard       : ${CMAKE_CXX_STANDARD}")
message(STATUS "OpenCV Version     : ${OpenCV_VERSION}")
message(STATUS "----------------------------------------------------")