# スペクトルからカラーチェッカーを計算

## 目的

* スペクトルからカラーチェッカーのRGB値を計算する
* 等色関数、D光源をサラッと扱える環境を整える

## 規格類

* 

## 準備

### 等色関数

```python
from colour.colorimetry import STANDARD_OBSERVERS_CMFS
from colour.colorimetry.spectrum import SpectralShape

CIE1931 = 'CIE 1931 2 Degree Standard Observer'
CIE1964 = 'CIE 1964 10 Degree Standard Observer'
CIE2015_2 = 'CIE 2012 2 Degree Standard Observer'
CIE2015_10 = 'CIE 2012 10 Degree Standard Observer'

shape = SpectralShape(500, 560)
print(STANDARD_OBSERVERS_CMFS[CIE1931].trim(shape))
```