# RRT+ODTを実行する 1DLUT & 3DLUT の作成(一部失敗)

## 目的

 ACES の RRT+ODT を実行する 1DLUT & 3DLUT を自作する。

## 背景

RRT+ODT を実行する 1DLUT & 3DLUT は既に Sony Picture Imageworks が公開している[1]。しかし、ここで公開されている LUT は 1DLUT で使用する ```minExposure``` や ```maxExposure``` といったパラメータが固定化されており、細かな精度評価を行う場合には少々不便である。ということで自由にパラメータ設定できる 1DLUT & 3DLUT を自作することにした。

## 結論



## 下準備

初めに、作成した 1DLUT & 3DLUT が正しいかを判断するための reference となる実行結果を用意することにした。用意したものを図1に示す。本記事の後半で 1DLUT & 3DLUT での変換結果が図1 と一致するかの判定を行う。

![99](./images5/99_src_bt2020_to_ap0_RRT_ODT.Academy.sRGB_100nits_dim.png)

<figure class="figure-image figure-image-fotolife" title="図1. reference となるテストパターン">[f:id:takuver4:20190805003045p:plain]<figcaption>図1. reference となるテストパターン</figcaption></figure>

図1 の作り方を簡単に説明する。以下の4ステップで作成した。

1. OETF: SMPTE ST2084(PQカーブ)、Gamut: BT.2020, White: D65 のテストパターンを作成。
2. テストパターンに ST2084 の EOTF を適用して Linear値に変換。
3. テストパターンの Gamut を BT.2020 --> AP0 に変換（RRTは AP0 を前提としているため）。
4. ```ctlrender```[2] を使用してテストパターンに RRT+ODT を適用。16bit TIFF で保存。

各ステップでのテストパターン様子を図2に示す。また、手順4. で使用した ```ctlrender``` の使い方は以下の過去記事を参照すること[3]。

[https://trev16.hatenablog.com/entry/2019/06/01/155644:embed:cite]

|  |  |
|:-----:|:-----:|
| ![1](./images_blog/00_org_bt2020_pq_d65.png) | ![2](./images_blog/01_linear_bt2020.png) |
| ![3](./images_blog/02_linear_bt2020_to_ap0.png) | ![4](./images_blog/03_rrt_odt.out.png) |

<div style="text-align: center;">図2. 各ステップの様子。左上、右上、左下、右下の順に Step1～4が並んでいる。</div>
|  |  |
|:-----:|:-----:|
|    [f:id:takuver4:20190806004336p:plain]    |     [f:id:takuver4:20190806004347p:plain]    |
|   [f:id:takuver4:20190806004358p:plain]     |    [f:id:takuver4:20190806004411p:plain]     |

## 1DLUT の作成

1DLUT は以下の記事で述べた通り、3DLUT 適用時の低階調の精度低下を防ぐために使用する。

[https://trev16.hatenablog.com/entry/2019/06/20/010121:embed:cite]

今回はソース画像が 10000nits すなわち Linear値で 100.0 の値を持っていたため、```maxExposure = 10.0 ``` とした。すなわち Linear 値で [tex: { 0.18 \times 2^{10} = 184.32 }] までの値を 3DLUT を使った変換の対象範囲とした。また、最小値に関しては ```minExposure = -10.0 ``` とした。これは ```midgray``` が Log2空間で ```0.5``` に割り当てるためである。

## 3DLUT の作成

## 参考資料

[1] imageworks, "OpenColorIO-Configs", https://github.com/imageworks/OpenColorIO-Configs

[2] ampas, "ctlrender", https://github.com/ampas/CTL/tree/master/ctlrender

[3] toruのブログ, "CTLで記述された ACES の RRT と ODT を画像に適用する", https://trev16.hatenablog.com/entry/2019/06/01/155644