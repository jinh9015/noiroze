import logging
import os
import boto3
import tempfile
import mysql.connector as connector
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import librosa
import numpy as np
import tensorflow as tf
import sys

os.environ[ 'NUMBA_CACHE_DIR' ] = '/tmp/'

model = None  # 모델 변수를 전역으로 정의합니다.

def load_and_configure_model():
    global model
    print("Exist model.hdf5 in the current working directory:", os.path.isfile(os.path.join(os.getcwd(), "model.hdf5")))
    model = load_model("model.hdf5", custom_objects={"Adam": Adam})
    print("Model loaded and configured.")

load_and_configure_model()  # 최초 한 번만 모델을 불러옵니다.

def lambda_handler(event, context):

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    print("Working directory:", os.getcwd())
    print("Exist app.py:", os.path.isfile(os.path.join(os.getcwd(), "app.py")))
    print("Exist model.hdf5:", os.path.isfile(os.path.join(os.getcwd(), "model.hdf5")))
    print("Exist requirements.txt:", os.path.isfile(os.path.join(os.getcwd(), "requirements.txt")))

    sys.stdout.flush()

    # 파일 및 버킷 이름 추출
    bucket_name = event['Records'][0]['s3']['bucket']['name']
    file_name = event['Records'][0]['s3']['object']['key']
    logger.debug(f"Bucket: {bucket_name}, Filename: {file_name}")

    # S3 파일 다운로드
    s3 = boto3.client('s3',
                    region_name='ap-northeast-2',
                    aws_access_key_id='AKIA5VZTIAOJVOZUKHWW',
                    aws_secret_access_key='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')
    logger.debug("Creating S3 client...")

    with tempfile.NamedTemporaryFile(dir='/tmp', suffix='.wav') as tmpfile:
        local_file = tmpfile.name  # 파일을 임시 디렉토리에 저장합니다.
        s3.download_file(bucket_name, file_name, local_file)
        logger.debug("Downloading file from S3 bucket...")

        process_audio(local_file, file_name)

def process_audio(file_path, file_name):
    file_name = os.path.basename(file_name)  # 파일 경로에서 순수한 파일 이름만 추출
    y, sr = librosa.load(file_path)
    print("Loading audio file...")

    class_names = ['충격음','가구끄는소리','악기소리','반려동물']

    check = [0,0,0,0] #가장 많이 나온것 판단

    n_fft = 2048
    hop_length = 512
    n_mels = 128
    fmin = 20
    fmax = 8000
    duration = 5 

    # 이미지화 -> 판단
    print("Processing audio...")

    for j in range(0, len(y)-duration*sr+1, duration*sr):
        y_interval = y[j:j+duration*sr]
        db = librosa.core.amplitude_to_db(librosa.feature.rms(y=y_interval), ref=10**0.7)[0][0]
        if -db <= 30:
            continue
        S = librosa.feature.melspectrogram(y=y_interval, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax)
        S_dB = librosa.power_to_db(S, ref=np.max)
        S_dB_norm = (S_dB - np.min(S_dB)) / (np.max(S_dB) - np.min(S_dB)) * 255
        # 이미지를 4차원으로 확장
        S_dB_norm_resized_4d = tf.expand_dims(S_dB_norm, axis=0)
        S_dB_norm_resized_4d = tf.repeat(S_dB_norm_resized_4d, 1, axis=-1).numpy()
        preds = model.predict(S_dB_norm_resized_4d)
        check[np.argmax(preds[0])]+=1

    print("Audio processing complete.")

    # 추론 결과값 출력
    result_class = class_names[np.argmax(check)]
    print("Result: " + result_class)

    # MySQL db에 결과값 저장

    # MySQL db 로그인 정보
    mydb = connector.connect(
        host="noiroze-db.csrccogv80xg.ap-northeast-2.rds.amazonaws.com",
        user="admin",
        password="admin123",
        database="noise_db",
        port=3306
    )
    print("Connecting to database...")

    try:
        mycursor = mydb.cursor()
        # MySQL db에서 place, created_at, dong, ho, value 값 가져오기
        mycursor.execute("SELECT place, created_at, dong, ho, value FROM main_sound_file WHERE file_name = %s", (file_name,))
        result = mycursor.fetchone()
        if result:
            place, created_at, dong, ho, value = result
        else:
            print("No record found for the given file_name")
            dong, ho, place, value, created_at = '102', '102', '안방', '44', '2022-02-22 22:22:22'

        # main_sound_level_verified 테이블에 값을 저장
        sql = "INSERT INTO main_sound_level_verified (file_name, sound_type, place, created_at, dong, ho, value) VALUES (%s, %s, %s, %s, %s, %s, %s)"
        val = (file_name, result_class, place, created_at, dong, ho, value)

        mycursor.execute(sql, val)

        mydb.commit()

        print(mycursor.rowcount, "record inserted.")
        
    except Exception as e:
        print("An error occurred:", e)

