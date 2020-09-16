# 네이버 영화평 감정분석
## 목표
- 네이버 영화평 데이터를 받아와 긍정 혹은 부정으로 감정을 분류하는 모델을 훈련하고, 새로운 영화평들의 감정을 분류합니다.
## 설명
- 인원 : 1명
- 작업툴 : Python, Tensorflow, Numpy, matplotlib, colab등
## 데이터
- [해당 깃허브](https://github.com/e9t/nsmc/)에서 ratings_test.txt와 ratings_train.txt을 다운받습니다.
## 참고자료
- [딥러닝을 이용한 자연어 처리 입문](https://wikidocs.net/book/2155)
## 프로그래밍
## 1. 라이브러리 import 하기
```python
 %matplotlib inline
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
```
```python
import pandas as pd
import tensorflow as tf
import numpy as np
```
```python
! sudo apt-get install g++ openjdk-7-jdk #Install Java 1.7+ 
#!sudo apt-get install python-dev; pip install konlpy # Python 2.x 
!sudo apt-get install python3-dev; pip3 install konlpy # Python 3.x 
!sudo apt-get install curl 
! bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)
```
```python
from konlpy.tag import Mecab
mecab = Mecab()
```
- Colab에 한글 형태소 분석기 KoNLPy를 설치합니다.

## 2. 데이터 불러오기
```python
from google.colab import drive
drive.mount('/content/gdrive')
```
- train셋과 test셋을 다운받고 구글 드라이브에 올려줍니다.
- Colab에 구글 드라이브를 마운트합니다.
```python
train = pd.DataFrame(pd.read_csv('/content/gdrive/My Drive/NLP/네이버영화리뷰실습/ratings_train.txt', sep='\t', quoting=3,encoding='utf-8')) 
train.dropna(inplace=True)
train.reset_index(inplace=True)
```
```python
test = pd.DataFrame(pd.read_csv('/content/gdrive/My Drive/NLP/네이버영화리뷰실습/ratings_test.txt', sep='\t', quoting=3,encoding='utf-8')) 
test.dropna(inplace=True)
test.reset_index(inplace=True)
```
- txt 파일을 csv로 읽어 옵니다.
- 결측치는 행 제거한 후, 인덱스를 재정렬합니다.
```python
print('train :',len(train))
print('test :', len(test))
```
- train셋은 149995개, test셋은 49997개의 리뷰가 존재합니다. (결측치 처리 이후의 값)
## 3. 데이터 전처리
### 3.1. 데이터 정제
```python
import re       # 정규표현식
remove_except_ko = re.compile(r'[^가-힣ㄱ-ㅎㅏ-ㅣ|\\s]')

def preprocess(text):
  text = re.sub(remove_except_ko,' ',text).strip()  # sub = replace
  return text

train['document'] = train['document'].map(lambda x : preprocess(x))
test['document'] = test['document'].map( lambda x : preprocess(x))
```
- 정규표현식을 사용하여 한글 이외의 글자들은 제외합니다.
### 3.2. 토큰화 및 불용어처리
```python
stop_word = ['께서','에서','이다','에게','으로','이랑','까지','부터','하다']
def postagging_mecab(text):
  text = mecab.morphs(text)
  text = [ i for i in text if len(i)>1]
  text = [ i for i in text if i not in stop_word]
  return text
  ```
  ```python
  def make_tokens(df):
  df['tokens']=''
  tokens_list = []
  for i, row in df.iterrows():
    token = postagging_mecab(df['document'][i])
    tokens_list.append(token)
    df['tokens'][i] = token
  return tokens_list, df
  ```
  - mecab을 사용하여 해당 영화평의 형태소만 가져옵니다.
  - for문을 사용하여 형태소만 담겨있는 list와 모두 합쳐진 데이터프레임을 출력하는 함수를 정의합니다.
  ```python
  train_list, train_df = make_tokens(train)
  test_list, test_df = make_tokens(test)
  ```
  

