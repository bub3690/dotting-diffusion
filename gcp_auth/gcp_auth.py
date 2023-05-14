### 환경 변수 설정

import os

# 사전에 json 파일을 다운로드 받을것!

# 역슬래쉬를 슬래쉬로 다 변경할 것
KEY_PATH = "D:/project/dotting_ai/gcp_auth/auth.json" # change to your key path.

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]= KEY_PATH