import os, cv2, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import features

# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------
def save_heatmap(array, title, filename, cmap='jet'):
    plt.imshow(array, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    os.makedirs('results', exist_ok=True)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def save_keypoints(image, keypoints, filename):
    vis = image.copy()
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(vis, (x, y), 2, (0,255,0), -1)
    os.makedirs('results', exist_ok=True)
    cv2.imwrite(filename, vis)

# -------------------------------------------------------------------
# 0️⃣ Load Images
# -------------------------------------------------------------------
img1 = cv2.imread('resources/yosemite1.jpg')
img2 = cv2.imread('resources/yosemite2.jpg')

gray1 = cv2.cvtColor(img1.astype(np.float32)/255.0, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2.astype(np.float32)/255.0, cv2.COLOR_BGR2GRAY)

# -------------------------------------------------------------------
# 1️⃣ Feature Computation (TODO1~6)
# -------------------------------------------------------------------
HKD = features.HarrisKeypointDetector()
SFD = features.SimpleFeatureDescriptor()
MFD = features.MOPSFeatureDescriptor()

# TODO1
a1, b1 = HKD.computeHarrisValues(gray1)
a2, b2 = HKD.computeHarrisValues(gray2)

# TODO3
d1 = HKD.detectKeypoints(img1)
d2 = HKD.detectKeypoints(img2)

# Filter weak keypoints
d1 = [kp for kp in d1 if kp.response > 0.01]
d2 = [kp for kp in d2 if kp.response > 0.01]

# TODO4~6
desc_simple_1 = SFD.describeFeatures(img1, d1)
desc_simple_2 = SFD.describeFeatures(img2, d2)
desc_mops_1 = MFD.describeFeatures(img1, d1)
desc_mops_2 = MFD.describeFeatures(img2, d2)

# -------------------------------------------------------------------
# 2️⃣ Visualization (TODO1, TODO3)
# -------------------------------------------------------------------
save_heatmap(a1, "Image1 - TODO1 Harris Strength", "results/img1_TODO1_harris_strength.png")
save_heatmap(a2, "Image2 - TODO1 Harris Strength", "results/img2_TODO1_harris_strength.png")

save_keypoints(img1, d1, "results/img1_TODO3_detected_keypoints.png")
save_keypoints(img2, d2, "results/img2_TODO3_detected_keypoints.png")

print("✅ Saved TODO1 & TODO3 visualizations.")

# -------------------------------------------------------------------
# 3️⃣ Matching (TODO7 - SSD, TODO8 - Ratio)
# -------------------------------------------------------------------
matcher_ssd = features.SSDFeatureMatcher()
matcher_ratio = features.RatioFeatureMatcher()

# ------------------------------
# TODO7 - SSD matching
# ------------------------------
# Step 1. SSD matcher를 이용해 두 이미지의 MOPS 디스크립터 매칭을 수행하시오.
matches_ssd = matcher_ssd.matchFeatures(desc_mops_1, desc_mops_2)

# Step 2. 거리(distance)를 기준으로 정렬하여 상위 150개의 매칭만 선택하시오.
matches_ssd = sorted(matches_ssd, key=lambda x: x.distance)[:150]

# Step 3. 매칭 결과를 시각화하여 PNG로 저장하시오.
ssd_vis = cv2.drawMatches(
    img1, d1, img2, d2, matches_ssd, None,
    matchColor=(0,255,0), singlePointColor=(255,0,0),
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
cv2.imwrite("results/TODO7_SSD_matches.png", ssd_vis)
print("✅ TODO7 (SSD) match result saved → results/TODO7_SSD_matches.png")

# ------------------------------
# TODO8 - Ratio matching
# ------------------------------
# Step 1. Ratio matcher를 이용해 두 이미지의 MOPS 디스크립터 매칭을 수행하시오.
matches_ratio = matcher_ratio.matchFeatures(desc_mops_1, desc_mops_2)

# Step 2. distance를 기준으로 정렬하여 상위 150개의 매칭만 선택하시오.
matches_ratio = sorted(matches_ratio, key=lambda x: x.distance)[:150]

# Step 3. 매칭 결과를 시각화하여 PNG로 저장하시오.
ratio_vis = cv2.drawMatches(
    img1, d1, img2, d2, matches_ratio, None,
    matchColor=(0,255,0), singlePointColor=(255,0,0),
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
cv2.imwrite("results/TODO8_Ratio_matches.png", ratio_vis)
print("✅ TODO8 (Ratio) match result saved → results/TODO8_Ratio_matches.png")

print("🎯 All TODO1–8 visualizations done! Files saved in 'results/'")


# --------------------------------------------------------------------------------------------------------------------------------------
# TODO 8의 매칭이 더 잘된 이유 (features.py 기반 상세 분석)
# --------------------------------------------------------------------------------------------------------------------------------------
#
# TODO 8 (RatioFeatureMatcher)의 매칭 결과가 TODO 7 (SSDFeatureMatcher)보다 훨씬 정밀하고 올바른 매칭(True Positive)이 많은 이유는,
# 매칭의 "신뢰도" 또는 "독보성(Discriminability)"을 평가하는 Lowe의 Ratio Test 방식을 사용했기 때문이다.
#
# features.py의 두 매칭 함수는 근본적인 차이점을 가진다.
#
# 1. TODO 7 (SSDFeatureMatcher)의 작동 방식과 한계
#
#   `features.py`의 `SSDFeatureMatcher.matchFeatures` 함수는 다음과 같이 동작한다.

#       a. `scipy.spatial.distance.cdist`를 이용해 desc1의 모든 특징점(N개)과 desc2의 모든 특징점(M개) 간의 유클리드 거리(SSD) 행렬 (N x M)을 계산한다.
#       b. `desc1`의 각 특징점 `i`에 대해, `dist[i]` 행에서 `np.argmin(dist[i])`를 호출하여 *단 하나의* 가장 가까운 특징점(min_dist)을 찾는다.
#       c. `match.distance`에 이 최소 거리(SSD1) 값을 그대로 저장한다.
#
#   `tests.py`에서는 `sorted(..., key=lambda x: x.distance)[:150]`를 통해 이 `match.distance`가 가장 작은 상위 150개를 선택한다.
#
# 2. TODO 7의 [문제점]
#   이 방식은 '가장 가까운' 매칭을 "맹목적"으로 선택한다.
#   만약 `desc1`의 한 특징점이 애매한(ambiguous) 특징점(예: 반복되는 창문 모서리)이라서 `desc2`에 있는 여러 특징점과 거의 비슷한 거리(예: 0.1, 0.11, 0.12)를 가진다면,
#   SSD 방식은 이 중 가장 작은 0.1을 선택한다.
#   하지만 2순위(0.11)와 차이가 거의 없으므로 이 매칭은 신뢰할 수 없다. (False Positive)
#
#   결국 TODO 7의 결과 이미지는 신뢰할 수 없는 애매한 매칭(False Positive)들이 단순히 SSD 값이 낮다는 이유로 상위 150개에 포함되어, 시각적으로 매우 지저분하게 나타난다.
#
#
# 3. TODO 8 (RatioFeatureMatcher)의 작동 방식과 우수성
#
#   `features.py`의 `RatioFeatureMatcher.matchFeatures` 함수는 다음과 같이 동작한다.
#
#       a. SSD 방식과 동일하게 `cdist`로 전체 거리 행렬을 계산한다.
#       b. `desc1`의 각 특징점 `i`에 대해, `dist[i]` 행에서 `np.argsort(dist[i])`를 호출하여 거리가 가까운 순서대로 정렬된 인덱스를 얻는다.
#       c. 이 인덱스를 이용해 1순위 최소 거리(SSD1 = dist[i, sort_Idx[0]])와 2순위 최소 거리(SSD2 = dist[i, sort_Idx[1]])를 모두 구한다.
#       d. `match.distance`에 실제 거리가 아닌, 두 거리의 비율 (Ratio = SSD1 / SSD2)을 저장한다.
#
# 4. TODO 8의 메인 차이점
#   `tests.py`에서 `sorted(..., key=lambda x: x.distance)[:150]`를 실행할 때, `x.distance`는 이제 "비율(Ratio)"을 의미한다.
#
#   Case 1: 좋은 매칭 (독보적인 매칭)
#       `desc1`의 특징점이 매우 독특(distinctive)하여 `desc2`의 올바른 짝과 매우 가깝고, 다른 모든 짝과는 거리가 멀다.
#       (예: SSD1 = 0.1, SSD2 = 0.8)
#       -> `match.distance` (Ratio) = 0.1 / 0.8 = 0.125 (매우 낮음)
#
#   Case 2: 나쁜 매칭 (애매한 매칭)
#     `desc1`의 특징점이 애매하여, `desc2`의 (틀린) 1순위 짝과 (틀린) 2순위 짝이 모두 비슷하게 가깝다.
#       (예: SSD1 = 0.1, SSD2 = 0.11)
#       -> `match.distance` (Ratio) = 0.1 / 0.11 = 0.909 (매우 높음)
#
# 5. 결론적으로
#   `tests.py`에서 `distance` (비율)를 기준으로 정렬하여 상위 150개를 뽑으면,
#   Case 2 (애매한 매칭, 높은 비율 값)는 자연스럽게 순위가 뒤로 밀려나고,
#   Case 1 (독보적인 매칭, 낮은 비율 값)이 우선적으로 선택된다.
#
# 따라서 TODO 8은 "단순히 가까운" 매칭이 아니라 "2순위와 비교했을 때 압도적으로 가까운" 매칭, 즉 신뢰할 수 있는 매칭만을 선별한다.
# 이로 인해 TODO 7에서 보였던 수많은 False Positive가 효과적으로 제거되고훨씬 더 정확하고 안정적인 매칭 결과를 얻을 수 있다.
# --------------------------------------------------------------------------------------------------------------------------------------