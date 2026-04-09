import requests
import time

# Law Agent API 
url = "http://localhost:8000/api/ask"
user_input = input("법률 관련 질문을 입력해주세요: ")
payload = {"query": user_input}

try:
    response = requests.post(url, data = payload)
    response.raise_for_status()
    job_id = response.json().get("job_id")
    print(f"Job ID: {job_id}")
    print(f"Status: {response.json().get('status')}")

    status_url = f"http://localhost:8000/api/jobs/{job_id}"
    print("⏳ 에이전트가 응답을 준비 중입니다. 잠시만 기다려주세요...")

    while True:
        res = requests.get(status_url)
        result = res.json()
        status = result.get("status")

        if status == "done":
            print("\n✨ 답변 생성이 완료되었습니다!")
            print("-" * 50)
            print(f"답변: {result['result']['answer']}")
            print("-" * 50)
            break

        elif status == "error":
            print(f"\n❌ 에러 발생: {result.get('error')}")
            break

        else:
            print(f"현재 상태: {status}...", end = "\r")
            time.sleep(2) # 2초 기다렸다가 다시 질문

except Exception as e:
    print(f"\n❌ 서버 연결 오류 발생: {e}")