import pymysql

# DB 연결 설정
conn = pymysql.connect(
    host="127.0.0.1",
    user="root",         # 네 MySQL 계정
    password="1234",  # 실제 root 비밀번호로 바꾸기
    database="mask_system",
    charset="utf8mb4"
)

print("✅ MySQL 연결 성공!")

# 연결 종료
conn.close()
print("🔌 연결 종료 완료.")
