# Project Structure

.
|-- README.md
|-- __init__.py
|-- api
|   |-- __init__.py
|   |-- album_router.py
|   |-- real_time_router.py
|   |-- upload_router.py
|   `-- user_router.py
|-- database
|   |-- __init__.py
|   |-- crud.py
|   |-- database.py
|   |-- models.py
|   `-- schemas.py
|-- main.py
|-- models
|   |-- __init__.py
|   |-- anomaly_detector.py
|   |-- convert.py
|   |-- pts
|   |   `-- pytorch_model.pth
|   `-- rt_anomaly_detector.py
|-- templates
|   |-- album_detail.html
|   |-- album_list.html
|   |-- base.html
|   |-- frame.html
|   |-- login.html
|   |-- main.html
|   |-- real_time.html
|   |-- signup.html
|   |-- stream.html
|   |-- upload.html
|   `-- video.html
`-- utils
    |-- __init__.py
    |-- config.py
    |-- security.py
    `-- utils.py

# Description

- api: URL별 로직 구현

- database: 데이터베이스 관련 설정 및 함수

- models: VAD 관련 모델

- templates: UI 템플릿. Bootstrap 사용

- utils: config 및 기타 함수