# Goyang_Public_Bicycle


# 고양시 공공자전거 스테이션 최적 위치 선정
### 배경
- 고양시는 2010년부터 공공자전거 서비스(피프틴, https://www.fifteenlife.com)를 도입하여
약 161개 스테이션(자전거 보관소)과 1,700여대의 공공자전거로 시민들에게 서비스를 제공 중에 있습니다.
- 현재 고양시는 신규 택지개발 등으로 도시화 지역이 늘어나고
인구 증가 등으로 인하여 기존 스테이션 위치에 대한 조정이 필요합니다.
### 목적
- 공공자전거 운영이력 데이터 및 공간 데이터를 활용하여 자전거 스테이션의 최적 위치를 선정하여
향후 시민들의 공공자전거(공유자전거) 사용에 대한 접근성을 개선하고자 합니다.
### 해결 과제
- 고양시에 대하여 공공자전거 스테이션의 최적위치를 제시하여 주시기 바랍니다.
- 분석 조건
    + 스테이션의 개수는 최대 300개소
    + 스테이션별 자전거 수용 용량 최대 30대
### 제출 방법
- 소스 코드는 COMPAS Notebook 에서 ipynb 파일을 다운로드하여 제출합니다.
- 소스 코드는 사용 패키지를 포함하여 오류 없이 실행되어야 합니다.
- 코드와 주석은 모두 UTF-8 을 사용하여야 합니다.
### 제출 대상
- 소스 코드 (필수) : ipynb 파일
  분석 결과(고양시 공공자전거 위치정보 및 스테이션별 수량)가 아래 예와 같이 출력될 수 있어야 합니다.


| 스테이션 번호 | 거치대수량 | X좌표(위도) | Y좌표(경도) |
|---------------|------------|-------------|-------------|
| 1             | 10         | 37.65       | 126.83      |


- 분석 보고서 (필수) : PPT 파일
  참여자는 제출한 모델을 기반으로 해당 지자체에서 필요시 모델링(시스템화)를 할 수 있도록
  데이터 정제/가공, 모델링 알고리즘 전체 결과 도출 과정을 상세히 설명해야 합니다.
  
  
| 분석 보고서 양식 | 분석 보고서 필수 기재사항 |
|---------------|------------|
|자유 양식 |데이터 전처리 및 분석과정 설명|
|분량 제한 없음||
|파워포인트(PPT) 형식으로 제출||
|필요시 설명자료(한글, 워드파일 등)를||
|중간 결과물에 포함하여 추가 제출 가능||
>기타 중간 결과물 : 오픈소스 툴(QGIS 등)을 이용한 중간 결과물들은 하나의 zip 파일로 압축하여 제출하십시오.
※ 파일 명에 대한 제약 사항은 없습니다.

### 분석 보고서 작성 요령
- 분석 개요
> 분석 배경
 과제관련 데이터, 이슈 등을 조망하여 전체적인 현황, 문제점 및 분석 계획(시나리오 등)을 간략히 서술합니다.\
 과제 분석에 소요된 기간(문서화 시간 제외)을 제시하여야 합니다.\
 예) 1일 2시간 씩 30일 소요
 
- 분석 목적
>과제에 대한 문제를 정의하고, 문제에 대한 해결 방향(분석방향)과 타당성을 정리하여 기술합니다.\
분석 절차 및 수행 방법\
주어진 데이터를 어떻게 무엇을 위해 사용하였는지 설명합니다.\
제공된 데이터를 cleansing(정제/가공)시 사유와 새로운 가공 파일 생성 시 사용된 가공 방법\
(예: A칼럼, B칼럼을 연산하여 C칼럼 생성 등)과 결과형식을 (포맷 등) 설명해야 합니다.\
각 분석 단계별 분석방법, 분석된 결과의 시사점을 설명해야 합니다.\
모델링 각 단계별 사용한 알고리즘과 해당 알고리즘 사용 이유, 모델링 결과, 이론적 근거를 제시해야 합니다.

- 분석 결과
>최종 모델링 된 결과에 대해 시각화를 진행하고, 이에 대한 설명(필요시)을 기술하여야 합니다.\
과제를 수행하면서 아쉬웠던 사항, 개선점, 추가 필요사항 등 제약사항에 대해 기술해야 합니다.\
제출 횟수 및 의무

‑ 결과 제출 횟수 무제한입니다.
- 소스파일, 분석 보고서는 필수이며, 누락시 감점 처리됩니다.
- 대회 종료 후 수상작은 COMPAS 게시판에 소스코드 및 분석보고서가 공개됩니다.
※ 수상자는 수상 소감 및 분석모델 설명을 위한 동영상 제작 시 협조하여야 합니다.

### 기타 참고 사항
- 제출된 모든 결과물의 사용권은 COMPAS 에 있습니다.
- 공간 분석을 위해 오픈소스 툴의 활용 결과물을 사용할 수 있습니다.
※ 예 : QGIS 등을 이용한 공간 분석 결과물을 COMPAS Notebook 에서 활용 가능합니다.

### 참고
지식 자료실 / [공간분석 활용](https://compas.lh.or.kr/gis)의 COMPAS 게시물들을 참조하여 도움을 얻을 수 있습니다.

### 대회사이트
- [compas](https://compas.lh.or.kr/) 

