# API 서버의 페이지네이션 적용을 위한 파일

from rest_framework.pagination import PageNumberPagination

class SetPagination(PageNumberPagination):
    page_size = 10                           # 페이지의 사이즈를 설정