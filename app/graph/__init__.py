"""LangGraph 기반 추천 오케스트레이터 패키지(실험 경로).

기존 파이프라인(``app/services/tasks/recommendation.py``)과 **별개의 실행 경로**입니다.
``RECOMMENDATION_LANGGRAPH=true`` 일 때만 임포트되므로, langgraph 미설치 환경에서
기존 경로는 영향을 받지 않습니다.
"""
