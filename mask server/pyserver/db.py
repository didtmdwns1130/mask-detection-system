# TRUNCATE TABLE events; - 테이블 구조는 남겨두고 데이터만 초기화하는 명령어

import pymysql

# MySQL 연결 설정
DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "1234",   # 실제 비번
    "database": "mask_system",
    "charset": "utf8mb4",
    "autocommit": True,   # True면 commit() 불필요
}

def save_event(status: str, prob: float, camera_id: str = "CAM01") -> None:
    """
    마스크 감지 이벤트 저장 (최소 필드만)
    - status: "MASK" 또는 "NO_MASK" (혹은 "MASK_ON"/"NO_MASK" 중 택1로 통일)
    - prob  : 0.0 ~ 1.0
    - camera_id: 카메라 식별자
    """
    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            sql = """
                INSERT INTO events (ts, status, prob, camera_id)
                VALUES (NOW(), %s, %s, %s)
            """
            cur.execute(sql, (status, float(prob), camera_id))
            new_id = cur.lastrowid
            print(f"[DB] 저장 완료 id={new_id}, status={status}, prob={prob:.2f}, cam={camera_id}")
    except Exception as e:
        print(f"[DB][ERR] {e}")
    finally:
        conn.close()
