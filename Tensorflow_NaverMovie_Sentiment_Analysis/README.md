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
