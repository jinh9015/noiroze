from django.contrib.auth import get_user_model            # 현재 활성화 된 유저 모델을 가져오는 함수 (현재 사용 X)
from django.contrib.auth import authenticate

from rest_framework import serializers
from common.models import CustomUser
from main.models import Sound_Level, Sound_File, Sound_Level_Verified, CommunityBoard, ComplainBoard, Reply, NoticeBoard
from rest_framework.authtoken.models import Token

class UserSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)  # 비밀번호 필드 추가

    class Meta:
        model = CustomUser
        fields = ('userid', 'password', 'email', 'name', 'apartment', 'dong', 'ho')     # API 서버에 보여질 필드

    def create(self, validated_data):                      # 유저 생성 시 호출되는 함수
        password = validated_data.pop('password')
        user = super().create(validated_data)
        user.set_password(password)                    # set_password는 Django의 AbstractBaseUser에서 제공하는 메서드로, 비밀번호를 안전하게 저장
        user.save()
        Token.objects.create(user=user)               # 사용자 생성 후 토큰 생성
        return user
    

class UserRegisterSerializer(serializers.ModelSerializer):
    password2 = serializers.CharField(write_only=True)

    class Meta:
        model = CustomUser             
        fields = ('userid', 'password', 'password2', 'email', 'name', 'apartment', 'dong', 'ho')         

    def validate(self, data):
        if data['password'] != data['password2']:
            raise serializers.ValidationError("비밀번호가 일치하지 않습니다.")
        return data

    def create(self, validated_data):
        validated_data.pop('password2')   # We do not need the password2 field anymore
        user = CustomUser.objects.create_user(
            validated_data['userid'],
            validated_data['password'],  # Here is the missing password argument
            validated_data['email'],
            validated_data['name'],
            validated_data['apartment'],
            validated_data['dong'],
            validated_data['ho'],
        )
        user.set_password(validated_data['password']) # set_password method를 사용하여 패스워드를 설정
        user.save() # 유저 저장 (common/models 의 UserManager를 통해 DB에 저장)
        return user

class UserLoginSerializer(serializers.Serializer):
    userid = serializers.CharField()
    password = serializers.CharField()                   # 로그인 시 필요한 필드들

    def validate(self, data):      # 로그인 검증하는 함수
        user = authenticate(userid=data['userid'], password=data['password'])
        if user and user.is_active:
            return user
        
        raise serializers.ValidationError("로그인 정보가 유효하지 않습니다.")
    

class SoundLevelSerializer(serializers.ModelSerializer) :
    class Meta :
        model = Sound_Level
        fields = ('dong', 'ho', 'place', 'value', 'created_at')

    def get_created_at(self, obj):
        return obj.created_at.strftime('%y_%m_%d_%H_%M') # 날짜를 년월일시분 까지표시, String형태
    

class SoundFileSerializer(serializers.ModelSerializer) :
    class Meta :
        model = Sound_File
        fields = ('dong', 'ho', 'place', 'value', 'file_name', 'sound_file', 'created_at')

    def get_created_at(self, obj):
        return obj.created_at.strftime('%y_%m_%d_%H_%M') # 날짜를 년월일시분 까지표시, String형태


class SoundLevelVerifiedSerializer(serializers.ModelSerializer) :
    class Meta :
        model = Sound_Level_Verified
        fields = ('dong', 'ho', 'place', 'value', 'created_at', 'sound_type')

    def get_created_at(self, obj):
        return obj.created_at.strftime('%y_%m_%d_%H_%M') # 날짜를 년월일시분 까지표시, String형태
    


class CommunityBoardSerializer(serializers.ModelSerializer):        # 커뮤니티 게시판 모델 직렬화
    author = serializers.SlugRelatedField(
        slug_field='userid',
        queryset=CustomUser.objects.all(),
    )

    class Meta:
        model = CommunityBoard
        fields = ['category', 'title', 'content', 'author', 'created_date', 'modify_date', 'like']
        read_only_fields = ('id',)

    # def get_created_date(self, obj):
    #     return obj.created_date.strftime('%y_%m_%d_%H_%M')        # 날짜를 년월일시분 까지표시, String형태
    
    def get_created_date(self, obj):
        return obj.created_date.strftime('%y_%m_%d_%H_%M')  # 날짜를 년월일시분 까지 표시, String 형태


class ReplySerializer(serializers.ModelSerializer):        # 커뮤니티 게시판 댓글 모델 직렬화
    author = serializers.SlugRelatedField(
        slug_field='userid',
        queryset=CustomUser.objects.all(),
    )

    class Meta:
        model = Reply
        fields = ['community_board', 'author', 'content', 'created_date', 'modify_date']
        read_only_fields = ('id',)
    
    def get_created_date(self, obj):
        return obj.created_date.strftime('%y_%m_%d_%H_%M')  # 날짜를 년월일시분 까지 표시, String 형태



class ComplainBoardSerializer(serializers.ModelSerializer):         # 민원접수 게시판 모델 직렬화
    author = serializers.SlugRelatedField(
        slug_field='userid',
        queryset=CustomUser.objects.all(),
    )

    class Meta:
        model = ComplainBoard
        fields = ['title', 'content', 'author', 'created_date']
        read_only_fields = ('id',)

    def get_created_date(self, obj):
        return obj.created_date.strftime('%y_%m_%d_%H_%M')  # 날짜를 년월일시분 까지 표시, String 형태
    

class NoticeBoardSerializer(serializers.ModelSerializer):         # 민원접수 게시판 모델 직렬화
    class Meta:
        model = NoticeBoard
        fields = ['title', 'content', 'created_date']

    def get_created_date(self, obj):
        return obj.created_date.strftime('%y_%m_%d_%H_%M')  # 날짜를 년월일시분 까지 표시, String 형태