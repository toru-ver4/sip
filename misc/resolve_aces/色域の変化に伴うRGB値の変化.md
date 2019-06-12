# 色域の変換に伴う Primary Colors の RGB値の変化

## 1. 目的

* Primary Colors を色域変換すると Primary Colors で無くなることを確認する(※) 
* 確認の際の可視化方法を工夫して少しでも分かりやすくする。

※例：Red を BT.709 → BT.2020 に変換すると、(1023, 0, 0) → (642, 71, 17) となり、Rチャネル以外にも値が生じる。
即ち Primary Colors では無くなる。

## 2. 背景

ACES の RRT関連の調査をしていて久々に恥ずかしい勘違いをしてしまったので、自戒の念を込めて記事を書くことにした。

## 3. 結論

Primary Colors を色域変換すると Primary Colors で無くなる。
各種色域の Primary Colors を ACES AP0 へ色域変換した結果を図1に示す。

## 4. 証明

愚直に R, G, B の Primary Colors を色域変換して確認した。
以下の順序で説明する。

* 色域変換のコード
* 色とRGBの比率と可視化
* Primary Colors の色域変換前後の可視化

### 4.1. 色域変換のコード

例によって Colour Science for Python[1] を使用する。```colour.RGB_to_RGB``` を使えば瞬殺である。
例を以下に示す(※1)。

※1 chromatic adaptation は "XYZ Scaling" を選択すること。"Bradford" や "CAT02" は色温度変換の計算で Primary Colors の値が変わってしまう([2]より自明)。

```python
>>> import numpy as np
>>> from colour import RGB_to_RGB
>>> from colour.models import BT709_COLOURSPACE, BT2020_COLOURSPACE
>>> chromatic_adapdation = "XYZ Scaling"
>>> bt709_red = np.array([1023, 0, 0]) / 1023
>>> ap0_red = RGB_to_RGB(bt709_red, BT709_COLOURSPACE, BT2020_COLOURSPACE, chromatic_adapdation)
>>> print(np.uint16(np.round(ap0_red * 1023)))
[642  71  17]
```

### 4.2. 色とRGBの比率と可視化

4.1. の計算結果を提示すれば証明は完了するのだが、少しでも結果を分かりやすく提示するために、可視化の方法を少々工夫しようと思う。

液晶ディスプレイのように**光の三原色**で表示する「色」は R, G, B の比率が大変重要である。
輝度成分(Y)を考えなければ、xy色度図上の色は R, G, B の比率で表現可能である。

例えば、図1のように D65 の白色点から Green の Primary への色変化に着目してみよう。
この時の No.0～No.5 の RGB値は表2となる。白色点から Green Priamry への変化では、
Red, Blue の割合が減少していることが読み取れる。

|No.|R|G|B|
|:---|---:|---:|---:|
|0|1023| 1023| 1023|
|1| 792| 1023|  792|
|2| 603| 1023|  603|
|3| 416| 1023|  416|
|4|   0| 1023|    0|

さて、表2は正確な値を知るには良いが R, G, B の**比率**を可視化する上では少々見づらい。そこで 100% 積み上げ棒グラフで表2の値を表現してみる。結果を図2に示す。




## 5. 参考文献

[1] Colour Science, Colour Science for Python, https://www.colour-science.org/

[2] Bruce Justin Lindbloom, Chromatic Adaptation, http://www.brucelindbloom.com/index.html?Eqn_ChromAdapt.html

