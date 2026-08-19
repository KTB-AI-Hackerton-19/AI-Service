"""테스트 공통 설정.

conftest 는 테스트 모듈보다 먼저 임포트되므로, 여기서 환경변수를 먼저 못박아
개발자의 ``.env`` 값이 테스트로 새어 들어오는 것을 막습니다.

테스트는 vLLM · Bedrock · S3 · Google · Tavily 어디에도 나가지 않아야 합니다.
외부에 나가면 CI 에서 깨지고, 로컬에서도 크레딧을 태우며, 무엇보다 느려집니다.
"""

import os

os.environ.setdefault("MODEL_BACKEND", "mock")
os.environ.setdefault("API_KEY", "test-key")
# 아래는 setdefault 가 아니라 강제입니다. .env 에 실제 값이 있어도 테스트에서는 꺼야 합니다.
os.environ["TAVILY_ENABLED"] = "false"
os.environ["TAVILY_API_KEY"] = ""
os.environ["GOOGLE_ACCESS_TOKEN"] = ""
os.environ["CALENDAR_AUTO_REGISTER"] = "false"
os.environ["ALLOW_PRIVATE_IMAGE_HOSTS"] = "false"
# .env 에 실제 Bedrock 키가 있어도 테스트가 AWS 로 나가지 않게 합니다.
os.environ["BEDROCK_API_KEY"] = "test-bedrock-token"
os.environ["AWS_BEARER_TOKEN_BEDROCK"] = "test-bedrock-token"
