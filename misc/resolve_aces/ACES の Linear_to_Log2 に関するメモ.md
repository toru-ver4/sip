# ACES の Linear to Log2 に関するメモ

## 目的

* ACES の Linear to Log2 変換を理解する

## 背景

知的欲求を満たすため ACES2065-1 のデータに RRT+ODT を適用する 3DLUT をカスタムで作ることになった。
一般に Linear のデータに対して 3DLUT を適用する場合は、低階調の精度低下を防ぐために[1] shaper と呼ばれる 1DLUT を事前に適用する。

OpenColorIO-Configs の aces_1.0.3 では shaper LUT の作成に ACES の "Linear to Log2", "Log2 to Linear" の CTL を利用している[2]。
この CTL はとても簡単な数式だが、In-Out の関係が筆者の予想と異なっていたので備忘録として記事を残す。

## 用語の定義

* Log2空間： ACES の CTL(ACESutil.Lin_to_Log2_param.ctl) を使って変換した後の空間
  * 一般的な用語ではない(と思う)。筆者が勝手に名付けた空間なので注意すること。

## 結論

* Log2空間は [0.0, 1.0] に正規化された空間である
* Linear --> Log2 への変換は2段階で行われる
  1. Log2 関数を使ったシンプルな変換（なお、```middleGray``` が ```0.0``` となるよう調整する）
  2. ```minExposure```, ```maxExposure``` がそれぞれ ```0.0```, ```1.0``` になるよう正規化
* ```minExposure```, ```maxExposure``` はそれぞれ、```middleGrey``` に対する露出を意味する。
  * 例:  ```maxExposure = 6.0``` の場合、Linear値で ```0.18 * (2^6.0) = 11.52``` が Log2空間の 1.0 に対応する
  * 例:  ```minExposure = -6.0``` の場合、Linear値で ```0.18 * (2^6.0) = 0.0028125``` が Log2空間の 0.0 に対応する
* ```middleGrey = 0.18```,  ```minExposure = -6.0```, ```maxExposure = 6.0``` の例を図1に示す。

![graph](./logNorm_with_log_x.png)

## 解説

ACES の Linear to Log2 変換は以下の関数で定義されている[3]。

```ctl
float lin_to_log2_32f
(
  float lin,
  float middleGrey,
  float minExposure,
  float maxExposure
)
{
  if (lin <= 0.0) return 0.0;

  float lg2 = log2(lin / middleGrey);
  float logNorm = (lg2 - minExposure)/(maxExposure - minExposure);

  if( logNorm < 0.0) logNorm = 0;

  return logNorm;
}
```

具体的な例を見るために ```lin_to_log2_32f()``` の ```lg2``` と ```logNorm``` をそれぞれプロットしてみる。パラメータはそれぞれ以下である。

* ```lin```: 2^(-6.5) ～ 2^(6.5)
* ```middleGrey```: 0.18
* ```minExposure```: -6.0
* ```maxExposure```: 6.0

図2に ```lg2``` のプロット結果を示す。

![graph2](./lg2_with_linear_x.png)

しかし、横軸が Linear スケールのため大変見づらい。横軸を対数スケールに変換したものを図3に示す。

![graph3](./log2_with_log_x.png)

図3を見ると分かる通り、```lg2``` の値は単純に ```log2(x * middleGrey)``` の計算を行っただけなので、[0:1] には正規化されていない。これを ```minExposure```, ```maxExposure``` に対応する値で [0:1] に正規化した値が ```logNorm```
すなわち Linear to Log2 変換の出力値である。プロットした結果は冒頭の図1で示した通りである。

## 感想

```minExposure``` と ```maxExposure```を (-6.0, 6.0) や (-10.0, 10.0) のように絶対値が同一となるように設定すると、Log2空間では ```middleGray``` が ```0.5``` となるため、```middleGray``` が基準であるとハッキリ分かり気持ちが良い。

ACES の RRT+ODT 以外でも shaper を作る機会があれば積極的に使いたいと思った。

## 参考資料

[1] OpenColorIO. How to Configure ColorSpace Allocation. https://opencolorio.org/configurations/allocation_vars.html (2019/06/15確認)

[2] OpenColorIO-Configs. aces.py. https://github.com/imageworks/OpenColorIO-Configs/blob/master/aces_1.0.3/python/aces_ocio/colorspaces/aces.py (2019/06/15確認)

[3] AMPAS. ACESutil.Lin_to_Log2_param.ctl. https://github.com/ampas/aces-dev/blob/master/transforms/ctl/utilities/ACESutil.Lin_to_Log2_param.ctl (2019/06/15確認)