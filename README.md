# 📸 SNS 게시글 분석 기반 신뢰할 수 있는 식당 정보 도출

![python](https://img.shields.io/badge/Python-3.9%2B-3776AB)
![selenium](https://img.shields.io/badge/Selenium-Web%20Automation-43B02A)
![bs4](https://img.shields.io/badge/BeautifulSoup-Parsing-4A86CF)
![pandas](https://img.shields.io/badge/Pandas-ETL-150458)
![mongodb](https://img.shields.io/badge/MongoDB-Optional-47A248)
![kakao](https://img.shields.io/badge/Kakao%20Map-REST%20API-FFCD00)
![openai](https://img.shields.io/badge/OpenAI-LLM%20Assist-412991)

> **요약**: 인스타그램 게시글을 **크롤링 → 광고/협찬성 필터링(규칙 + LLM 보조) → 주소·좌표 표준화(카카오맵 API)** 로 정제하여  
> **검증 가능한 맛집 데이터셋**을 생성합니다. 산출물은 **CSV** 중심이며, 옵션으로 **MongoDB**에도 적재할 수 있도록 하였습니다.

---

## 📌 프로젝트 개요
SNS에는 실제 후기와 광고·협찬 게시글이 섞여 있어 신뢰도 높은 맛집 선별이 어렵습니다. 이 프로젝트는 크롤링한 게시글에서 LLM으로 식당명과 주소를 정규 추출한 뒤, 추출된 후보에 대해 규칙 기반(해시태그·키워드·CTA)으로 광고 여부를 판별합니다. 비광고로 판정된 식당만 카카오맵 REST API로 주소를 표준화하고 위경도·행정동을 매핑하며, 오기재·지점 중복을 후처리하여 품질을 높였습니다. 최종 산출물은 CSV 파일과 MongoDB 두 형태로 저장하고, 산출 결과는 지도로 빠르게 확인할 수 있습니다.

---

## 🧾 프로젝트 내용 요약
- **크롤링**: 인스타그램 게시글에서 본문·해시태그·댓글·좋아요 등 핵심 신호를 안정적으로 모아 **원천 데이터셋**을 만듭니다.  
- **LLM 추출**: 수집한 텍스트에서 **식당명과 주소를 정규 필드로 구조화**하여 후보 리스트를 생성합니다(지번/도로명 표기 차이도 통일).
- **광고 판별(규칙 기반)**: 해시태그·키워드·CTA 등 규칙을 기반으로 광고 여부를 판정합니다.  
- **지도 표준화**: 카카오맵 API로 **주소 정규화 → 위경도·행정동 매핑**을 수행하고, 오기재·다점포는 **거리/문자 유사도**로 **중복 병합**합니다. 간이 지도 페이지를 생성하여 시각적으로 확인이 가능하도록 하였습니다.
- **저장**: 최종 결과는 **CSV**로 저장하고, 필요 시 **MongoDB**를 사용하여 운영 저장소에 적재할 수 있도록 하였습니다.  



---

## 🧠 핵심 기능

### 인스타그램 크롤링
- 로그인(환경변수 IG_ID/IG_PW) → 해시태그 검색 → 게시글 순회. 본문·해시태그·댓글·좋아요를 수집하고, 다중 셀렉터/정규식 백업과 실패 시 로그 후 건너뛰기로 수집을 끊기지 않게 유지합니다.
- 결과는 `data/region_restaurant.csv`에 `post_id` 기준 중복 제거로 저장합니다.

### 텍스트 전처리
- 이모지·URL·전화번호·과도한 해시태그를 마스킹/삭제하고 공백을 정리합니다.
- 리스트형 필드(예: 해시태그·댓글)는 JSON 문자열로 통일해 CSV에서 깨지지 않게 합니다(UTF-8-sig).

### LLM 기반 정보 추출
- 게시글에서 식당명·주소를 JSON 스키마로 추출합니다(모델: gpt-3.5-turbo).
- 요청이 실패하면 잠깐 대기했다가 최대 N회까지 다시 시도합니다(과도한 호출/일시적인 오류 대비).
- 추출된 후보는 리스트로 보존해 `data/extracted.csv`에 기록하고, 실패 건(Prompt 정책에 의한 실패 혹은 API 호출 실패)은 `data/extracted_failures.csv`에 남깁니다.

### 광고/협찬 판별(규칙 기반)
- 해시태그(#광고/#협찬) + 키워드(제공받음/sponsored/paid partnership 등) + CTA(예약/문의/DM/링크/전화) 신호를 점수화하여 is_ad(광고 여부) 라벨을 만듭니다.
- 임계값과 여러 모드(--mode=balanced|recall)를 통해 보수/공격적 라벨링을 선택할 수 있으며 결과는 `data/ad_extracted.csv`로 저장합니다.

### 주소 표준화 & 지도 생성
- 카카오 REST API로 도로명 주소 인식 → 위/경도 변환 후, 좌표 반경에서 키워드 재탐색으로 장소명을 정밀화합니다.
- 비광고(is_ad=false)만 표시한 인터랙티브 지도 `matched_places_map.html`을 생성하며, 표시 개수/중심/줌/중복 제거 키는 환경변수로 조절합니다.

### 결과 저장 & 운영 옵션
- 기본 산출물은 CSV(UTF-8-sig). 필요 시 MongoDB UPSERT로 운영 저장소에 적재(merge 키 기준 중복 없이 갱신).
- 파이프라인 중간 산출물은 `region_restaurant.csv` → `extracted.csv` → `ad_extracted.csv` → `clean/non_ads.csv`, `clean/ads.csv` 순으로 축적됩니다.

### 규칙 진단용 모델(선택 사용)
- TF-IDF(문자 n-gram)+숫자 피처 기반 LGBM/RF/XGB를 고정 시드 교차검증(OOF) 으로 학습해 Precision/Recall/F1/ROC-AUC를 산출합니다.
- 규칙 라벨 기준의 의심 샘플/경계 샘플을 `data/qc/*.csv`로 내보내 규칙 점검에 활용합니다. (진단 및 비교 목적)

---

## 🗃 데이터 스키마 (CSV 기준)

파이프라인 단계별 **CSV 컬럼 정의**입니다.\
CSV 인코딩은 **UTF-8-sig** 를 사용합니다.\
결측은 빈 문자열(`""`) 또는 `[]`로 표기하며, 기본 조인 키는 **`post_id`** 입니다.


### 1) 원천 수집 — `region_restaurant.csv`
인스타그램에서 직접 수집한 원본.

| 컬럼 | 타입 | 예시 | 설명 |
|---|---|---|---|
| `post_id` | string | `DN43NiUe...` | 게시글 고유 ID |
| `content` | string | `엄청 맛있..` | 본문 텍스트 |
| `likes` | int | `14` | 좋아요 수 |
| `hashtags` | string(JSON array) | `["#여수맛집"]` | 해시태그 목록 |
| `comments` | string(JSON array) | `["", "#여수맛집"]` | 댓글 |
| `date` | date(YYYY-MM-DD) | `2025-08-02` | 게시 날짜 |
| `search_tag` | string | `여수맛집` | 수집에 사용한 검색 태그 |

> 무결성: `post_id` 유일, `date`는 `YYYY-MM-DD` 형식 유지.

### 2) LLM 추출 — `extracted.csv`
원천에 **식당명/주소 후보**가 추가된 결과.

| 컬럼 | 타입 | 예시 | 설명 |
|---|---|---|---|
| *(원천 컬럼 전부)* |  |  | `region_restaurant.csv`의 모든 컬럼 유지 |
| `restaurant_name` | string \| string(JSON array) | `["식당명"]` | LLM이 추출한 식당명(복수 불가능) |
| `address` | string(JSON array) | `["식당 주소"]` | LLM이 추출한 주소 후보(복수 불가능) |

> 규칙: `restaurant_name`, `address`는 **후보군을 보존**하기 위해 리스트 문자열로 저장합니다. 후보가 없으면 `[]`.


### 3) 광고 판별 — `ad_extracted.csv`
LLM 추출 결과에 **광고 라벨**만 추가된 파일.

| 컬럼 | 타입 | 예시 | 설명 |
|---|---|---|---|
| *(extracted 컬럼 전부)* |  |  | `extracted.csv`의 모든 컬럼 유지 |
| `is_ad` | bool | `false` | 규칙 기반 판별 결과 |

> 참고: 현재 스키마는 `is_ad`만 추가합니다. 필요 시 `ad_reason`, `ad_keywords` 등을 후속 버전에서 확장 가능.


### 전환 요약
- `region_restaurant.csv` → **LLM 추출** → `extracted.csv`  
- `extracted.csv` → **규칙 기반 광고 판별** → `ad_extracted.csv`  

이후 단계(지도 표준화)에서 `ad_extracted.csv`의 **`is_ad=false` 행만** 사용해 주소 정규화·좌표 매핑을 수행합니다.


---
## 📈 규칙 기반 광고 판별 검증

**규칙에 기반하여 광고를 판별**한 뒤, 학습 모델(XGBoost / RandomForest / LightGBM)은
규칙이 잘 동작하는지 규칙 라벨('is_ad')을 기준으로 **진단·비교**하기 위한 용도로 사용합니다.

- **평가 설정**: **광고(`is_ad=1`)를 양성(positive) 클래스**로 두고 지표를 계산합니다.
- **방법**: `ad_extracted.csv`를 `seed=42`로 **Stratified K-Fold**(OOF) → XGB/RF/LGBM 학습 →  
  **Precision·Recall·F1·PR-AUC·ROC-AUC**를 계산하고 서로 비교합니다.

| Model        | Precision | Recall | F1   | ROC-AUC |
|:------------:|:-----------:|:-----------------:|:------:|:-----:|
| XGBoost      | 0.91             | 0.81  | 0.86 | 0.96    |
| RandomForest | 0.90             | 0.85  | 0.87 | 0.96    |
| LightGBM     | 0.96             | 0.81  | 0.88 | 0.96    |

> 이 표는 **규칙이 만든 라벨이 일관적으로 작동하는지**를 확인하기 위한 참고용입니다.

> 세 모델 모두에서 ROC-AUC가 **≈0.96**, F1이 **0.86–0.88**로 꾸준히 높게 나온 것은, **규칙(`is_ad`) 라벨을 모델이 잘 재현할 만큼 규칙이 일관적으로 적용**되고 있다는 뜻입니다.  
Precision(0.90–0.96)이 높은 편이어서 **“광고로 판정된 것의 정확도”가 높고**, Recall(0.81–0.85)도 양호하여 **규칙이 놓치는 광고 패턴이 일부 있지만 크지 않음**을 시사합니다.  
다만 이 평가는 **규칙 라벨과의 합치도**이며 절대적인 정답과의 정확도는 아닙니다. 따라서 주기적으로 **오분류 샘플을 점검**하여 해시태그/키워드/CTA 규칙을 보완하는 것이 좋습니다.

---

## 🗺️ 지도 스냅샷 (예시)

최종 CSV에서 **`is_ad=false`** 인 레코드만 사용해 생성한 간이 지도(`matched_places_map.html`)의 예시입니다.  
아래 이미지는 HTML을 캡처한 정적 스냅샷으로, 실제 파일은 로컬에서 열면 인터랙티브로 확인할 수 있습니다.

<img width="385" height="384" alt="예시" src="https://github.com/user-attachments/assets/3c511b26-ace1-43e4-969b-da5235cb2d35" /><br>

**용도**: 비광고로 확정된 맛집 후보를 공간적으로 한눈에 확인\
**파일**: `matched_places_map.html` (로컬 브라우저에서 열기)  

---

## 🧯 트러블슈팅

| 문제(이슈) | 내가 취한 접근 | 결과/효과 |
|---|---|---|
| DOM 변동으로 파싱 실패 | 다중 셀렉터·정규식 **백업 파서** + 실패 시 **건너뛰기(로그만 남김)** | 크롤러가 **중단 없이** 수집 지속 |
| 초기 규칙/전통 NLP로는 식당명·주소 추출 한계 | **LLM으로 식당명/주소 정규 추출**(JSON 스키마 강제) | 식당명·주소 **인식률 및 정확도↑**, 후속 단계 **안정화** |
| 주소 표준화·지점 중복 | 카카오맵으로 **정규화/좌표화** 후, **가까운 동일 지점 병합** | 지도 결과 **중복↓**, 위치 **정확도↑** |
| 광고 판별 혼선 | **해시태그/키워드/CTA 규칙 세트** 설계(우선순위/가중 적용) | 재현 가능한 `is_ad` 라벨, **오탐·누락 감소** |

---

## 🔧 환경 변수(.env 템플릿 예시)
```bash
# Kakao & OpenAI (필수)
KAKAO_REST_API_KEY=YOUR_KAKAO_KEY

OPENAI_API_KEY=YOUR_OPENAI_KEY
OPENAI_MODEL=YOUR_MODEL

# Mongo (옵션)
MONGO_URI=mongodb://localhost:27017
MONGO_DB=sns_eats  # DB명

# Instagram 계정
IG_ID=YOUR_INSTAGRAM_ID
IG_PW=YOUR_INSTAGRAM_PW

# 인스타그램 검색 hashtags(앞에 #붙여도 되고, 안 붙여도 됨)
IG_SEARCH_TAG=여수맛집

# 한 번에 긁을 게시글 개수
IG_CRAWL_COUNT=10

```

## 🚀 Quick Start
```bash

# 가상환경 생성 및 활성화 (수정)
conda env create -n snsa -f environment.yml
conda activate snsa

# 게시글 크롤링
python scripts/run_crawling_instagram.py

# LLM을 추출
python src/modules/llm/run_enrich_llm.py

# 규칙 기반 광고 판별
python scripts/make_ad_labels.py

# 광고와 비광고를 구분하여 저장   
python scripts/filter_for_map.py

# 비광고로 판단된 데이터를 바탕으로 지도 생성
python scripts/run_map_api.py

# MongoDB에 적재 (선택)  
python scripts/load_to_mongo.py           
```
---
<details>
 
<summary><h2>📁 폴더 구조</h2></summary>

```bash
SNS-EATS/
├─ artifacts/
├─ data/
├─ scripts/
│  ├─ filter_for_map.py
│  ├─ load_to_mongo.py
│  ├─ make_ad_labels.py
│  ├─ qc_nonad_quality.py
│  ├─ run_crawling_instagram.py
│  └─ run_map_api.py
├─ src/
│  ├─ modules/
│  │  ├─ db/
│  │  │  └─ db_mongo.py
│  │  ├─ llm/
│  │  │  ├─ llm_extract.py
│  │  │  └─ run_enrich_llm.py
│  │  ├─ ad_rules.py
│  │  ├─ crawling_instagram.py
│  │  ├─ data_io.py
│  │  └─ map_api.py
│  └─ utils/
│     ├─ crawling_utils.py
│     └─ map_utils.py
├─ environment.yml
├─ .env
└─ README.md
```
  
</details>

