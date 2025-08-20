# FlockAnalysis.msg
# 새떼 분석 결과를 종합적으로 담는 메시지

std_msgs/Header header

# --- 주요 분석 결과 ---
bool is_flock             # 새떼로 판단되는지 여부
string[] species          # 감지된 모든 새의 종류 (중복 제거)
int32 number_of_birds     # 감지된 새의 총 마리 수
string formation          # 군집 형태 (e.g., "V-shape", "clustered", "scattered")
int32 number_of_clusters  # 군집의 수
float32 density           # 밀도 (0.0 ~ 1.0)
geometry_msgs/Vector3 direction # 새떼의 평균 이동 방향 (선택적 기능)

# --- 개별 객체 정보 (기존 DetectedObjects와 유사) ---
int32 image_width         # 원본 이미지 너비
int32 image_height        # 원본 이미지 높이
bird_repellent_msgs/DetectedObject[] objects # 감지된 개별 새 객체 목록
geometry_msgs/Point[] critical_points # 군집의 주요 지점 (예: V-shape의 꼭짓점, Line의 양 끝점, Dispersed의 중심점)
