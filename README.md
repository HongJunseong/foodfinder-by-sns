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

## 📁 폴더 구조
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

---

## 🧠 핵심 설계 포인트

- **단계형 파이프라인**  
  데이터 수집 → LLM 추출 → 규칙 기반 광고 판별 → 지도 표준화 → 저장. 각 단계가 **독립 실행/재실행** 가능해 디버깅이 쉽습니다.

- **LLM을 통한 광고 1차 필터링 및 필수 정보 추출**
  LLM으로 게시글에서 **식당명·주소를 정규 형태로 추출**하면서, 문맥 기반으로 **광고 가능성이 낮은 후보를 1차 선별**합니다.

- **광고 규칙 집합**  
  게시글 내의 **해시태그(#광고·#협찬)**, **키워드(제공받음/체험단/sponsored)**, **CTA(예약/문의/DM/구매링크)** 등을 고려하여 광고를 판별합니다.

- **주소 표준화 & 중복 병합**  
  카카오맵 API로 **주소 정규화 → 위경도·행정동 매핑**을 수행하고, **거리 + 문자열 유사도**로 **지점 중복**을 정리합니다.

- **견고한 크롤러**  
  DOM 변동에 대비해 **다중 셀렉터/정규식 백업**을 두고, 실패 레코드는 **로깅 후 건너뜀**으로 **파이프라인 중단**을 막습니다.

- **즉시 검수 가능한 산출물**  
  결과를 `matched_places_map.html`로 내보내 **비(非)광고로 확정된 맛집 후보**를 지도로 빠르게 확인할 수 있습니다.

- **저장 전략 이중화**  
  기본은 **CSV**, 운영 탐색·검색이 필요하면 **MongoDB upsert**로 중복 없이 갱신합니다.

- **환경 기반 설정**  
  모든 키/옵션은 **.env**로 분리하여 코드와 **비밀정보**를 격리, 로컬↔운영 전환을 단순화했습니다.

---

## 🗃 데이터 스키마 (CSV 기준)

파이프라인 단계별 **CSV 컬럼 정의**입니다.\
CSV 인코딩은 **UTF-8-sig** 를 사용합니다.\
결측은 빈 문자열(`""`) 또는 `[]`로 표기하며, 기본 조인 키는 **`post_id`** 입니다.


### 1) 원천 수집 — `region_restaurant.csv`
인스타그램에서 직접 수집한 원본.

| 컬럼 | 타입 | 예시 | 설명 |
|---|---|---|---|
| `post_id` | string | `DN43NiUe...` | 게시글 고유 ID(조인 키) |
| `content` | string | `엄청 맛있..` | 본문 텍스트(이모지 가능) |
| `likes` | int | `14` | 좋아요 수(파싱 실패 시 빈값/0) |
| `hashtags` | string(JSON array) | `["#여수맛집"]` | 해시태그 목록 |
| `comments` | string(JSON array) | `["", "#여수맛집"]` | 댓글 일부/요약 |
| `date` | date(YYYY-MM-DD) | `2025-08-02` | 게시 날짜(현지 기준) |
| `search_tag` | string | `여수맛집` | 수집에 사용한 검색 태그 |

> 무결성: `post_id` 유일, `date`는 `YYYY-MM-DD` 형식 유지.

### 2) LLM 추출 — `extracted.csv`
원천에 **식당명/주소 후보**가 추가된 결과.

| 컬럼 | 타입 | 예시 | 설명 |
|---|---|---|---|
| *(원천 컬럼 전부)* |  |  | `region_restaurant.csv`의 모든 컬럼 유지 |
| `restaurant_name` | string \| string(JSON array) | `["식당명"]` | LLM이 추출한 식당명(복수 가능, 보통 1개) |
| `address` | string(JSON array) | `["식당 주소"]` | LLM이 추출한 주소 후보(복수 가능) |

> 규칙: `restaurant_name`, `address`는 **후보군을 보존**하기 위해 리스트 문자열로 저장합니다. 후보가 없으면 `[]`.


### 3) 광고 판별 — `ad_extracted.csv`
LLM 추출 결과에 **광고 라벨**만 추가된 파일.

| 컬럼 | 타입 | 예시 | 설명 |
|---|---|---|---|
| *(extracted 컬럼 전부)* |  |  | `extracted.csv`의 모든 컬럼 유지 |
| `is_ad` | bool | `false` | 규칙 기반 판별 결과(광고/협찬 여부) |

> 참고: 현재 스키마는 `is_ad`만 추가합니다. 필요 시 `ad_reason`, `ad_keywords` 등을 후속 버전에서 확장 가능.


### 전환 요약
- `region_restaurant.csv` → **LLM 추출** → `extracted.csv`  
- `extracted.csv` → **규칙 기반 광고 판별** → `ad_extracted.csv`  

이후 단계(지도 표준화)에서 `ad_extracted.csv`의 **`is_ad=false` 행만** 사용해 주소 정규화·좌표 매핑을 수행합니다.


---
## 📈 규칙 기반 판별 검증

**규칙 기반 광고 판별**을 사용하고, 학습 모델(XGBoost / RandomForest / LightGBM)은
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
