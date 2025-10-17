[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/MQ1nsMrd)
# 📘 CV_class_2025_2_Assignment_2

지난 시간에는 다소 어렵다는 피드백이 많아, 이번 과제는 보다 쉽게 구성했습니다.
이번 수업에서는 **Harris 코너를 이용해 이미지의 특징을 탐지하고**, **이미지 간 매칭(feature matching)** 을 수행하는 내용을 다룹니다.
천천히 과정을 따라가며 즐겨주시길 바랍니다. 😊

📖 **Harris 코너에 대한 추가 설명**은 아래 링크를 참고하세요.
👉 [Harris Corner 설명](https://songminkee.github.io/studyblog/computer%20vision/2020/06/22/harris.html)

---

## 🧩 Project 2: Feature Detection & Matching

이 프로젝트에서는 **이미지의 특징점을 탐지하고**, **두 이미지 간의 대응 관계를 찾는 과정**을 구현합니다.

### 주요 목표

1. Harris 알고리즘을 이용한 Feature Detection
2. Feature Descriptor 구현 (Simple & MOPS)
3. SSD 및 Ratio Test를 이용한 Feature Matching
4. 서로 다른 Descriptor 및 Matching 방법의 성능을 ROC Curve로 비교

---

## 🪜 단계별 진행

1. 소스 이미지의 각 픽셀에서 Harris Corner Strength 계산
2. 로컬 최대값(Local Maxima) 검출
3. **Simple Feature Descriptor**와 **MOPS (Multi-Scale Oriented Patches) Descriptor** 구현
4. **SSD** 또는 **Ratio Distance**를 이용한 Feature Matching 수행
5. 서로 다른 Descriptor와 Matching 함수의 성능을 **ROC Curve**로 비교

---

## 🧠 목표와 실행방법

`tests.py`의 **TODO 7, 8 단계**를 완성하고,
결과 이미지를 `print`하여 **시각화**하는 것이 이번 과제의 목표입니다.

과제의 구조는 매우 간단하며,
**함수의 동작 원리**와 **Harris 코너 검출의 개념**을 이해하고 있다면 쉽게 해결할 수 있습니다.

`features.py`에는 관련된 클래스와 함수들이 정의되어 있으며,
실행은 `tests.py` 파일을 실행하면 됩니다.

---

## 💡 예시 결과

**TODO 3 단계**를 시각화하면 다음과 같이 특징점들이 매핑됩니다.

<img width="640" height="480" alt="img1_TODO3_detected_keypoints" src="https://github.com/user-attachments/assets/1362b9d7-3ee1-463d-b406-613dcd9b0f92" /> 

<img width="640" height="480" alt="img2_TODO3_detected_keypoints" src="https://github.com/user-attachments/assets/25445431-5b8d-4085-bad3-0fdf9554ee3a" />  

---

**TODO 7단계**와 **TODO 8단계**를 수행하면, 아래와 같이 이미지 간 특징점 매칭 결과를 확인할 수 있습니다.

**SSD Matching 결과 (TODO 7)** <img width="1280" height="480" alt="TODO7_SSD_matches" src="https://github.com/user-attachments/assets/8e475728-20eb-4b77-8f1e-c6c8f3a5c350" />

**Ratio Matching 결과 (TODO 8)** <img width="1280" height="480" alt="TODO8_Ratio_matches" src="https://github.com/user-attachments/assets/513fec09-68e3-425f-a1f4-167bb9210c10" />

---

## ✏️ 마무리

TODO 7과 8 단계를 모두 완료하면 위와 같은 **매칭 결과(MATCHING)** 를 얻을 수 있습니다. 
마지막으로, **8단계의 매칭이 더 잘된 이유**를 구체적으로 **각주(설명)** 로 꼭 작성해 주세요. (해당 각주내용이 반영되어 있지 않으면 0점 처리하겠습니다.)

## 제출

1. tests.py 코드제출
2. TODO3,7,8 print 이미지 3장 제출
3. TODO 8단계의 매칭이 잘된 이유를 features.py코드를 바탕으로 구체적으로 작성(위치 : tests.py 코드  맨 마지막 줄에 작성)
