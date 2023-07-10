from django.urls import path, include
from rest_framework import routers
from .views import *
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)

app_name = 'api'

router = routers.DefaultRouter()
router.register('sound_level', SoundLevelViewSet)
router.register('sound_file', SoundFileViewSet)
router.register('sound_verified', SoundLevelVerifiedViewSet)
router.register('community_board', CommunityBoardViewSet)
router.register('community_board_reply', ReplyViewSet)
router.register('complain_board', ComplainBoardViewSet)
router.register('notice_board', NoticeBoardViewSet)

urlpatterns = [
    path('user_detail/', UserDetailView.as_view()),         # api/user_detail/   로 엔드포인트 설정. 로그인 한 유저 정보 확인 가능
    path('user_list/', UserListView.as_view()),             # api/user_list/   로 엔드포인트 설정. 유저 목록 확인 가능.
    path('user_register/', UserRegisterView.as_view(), name='user_register'),     # api/user_register/  회원가입 시 데이터 저장되는 엔드포인트
    path('user_login/', UserLoginView.as_view(), name='user_login'),           # api/user_login/     로그인 시도시, 자격증명 검증하는 엔드포인트
    path('user_logout/', UserLogoutView.as_view(), name='user_logout'),               # api/user_logout/  로그아웃 시, 토큰을 제거하는 엔드포인트
    path('token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),          # 사용자 토큰 생성
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),         # 사용자 토큰 리프레시

    path('', include(router.urls))
]



'''
router = routers.DefaultRouter()


router.register('user_list', UserListView.as_view()),
router.register('user_register', UserRegisterView.as_view(), name='user_register'),
router.register('user_login', UserLoginView.as_view(), name='user_login'),

urlpatterns = [
     path('', include(router.urls))
]
'''