# RRT+ODTを実行する 1DLUT & 3DLUT の作成(一部失敗)

## 目的

 ACES の RRT+ODT を実行する 1DLUT & 3DLUT を自作する。

## 背景

RRT+ODT を実行する 1DLUT & 3DLUT は既に Sony Picture Imageworks が公開している[1]。しかし、ここで公開されている LUT は 1DLUT で使用する ```min_exposure``` や ```max_exposure``` といったパラメータが固定化されており、細かな精度評価を行う場合には少々不便である。ということで自由にパラメータ設定できる 1DLUT & 3DLUT を自作することにした。

## 下準備

初めに、作成した 1DLUT & 3DLUT が正しいかを判断するための reference となる実行結果を用意することにした。用意したものを図1に示す。本記事の後半で 1DLUT & 3DLUT での変換結果が図1 と一致するかの判定を行う。

![99](./images5/99_src_bt2020_to_ap0_RRT_ODT.Academy.sRGB_100nits_dim.png)

<figure class="figure-image figure-image-fotolife" title="図1. reference となるテストパターン">[f:id:takuver4:20190805003045p:plain]<figcaption>図1. reference となるテストパターン</figcaption></figure>

図1 の作り方を簡単に説明する。以下の4ステップで作成した。

1. OETF: SMPTE ST2084(PQカーブ)、Gamut: BT.2020, White: D65 のテストパターンを作成。
2. テストパターンに ST2084 の EOTF を適用して Linear値に変換。
3. テストパターンの Gamut を BT.2020 --> AP0 に変換（RRTは AP0 を前提としているため）。
4. ```ctlrender```[2] を使用してテストパターンに RRT+ODT を適用。16bit TIFF で保存。

 手順4. で使用した ```ctlrender``` の使い方は以下の過去記事を参照[3]。

[https://trev16.hatenablog.com/entry/2019/06/01/155644:embed:cite]

## 1DLUT の作成

1DLUT は以下の記事で述べた通り、3DLUT 適用時の低階調の精度低下を防ぐために使用する。

[https://trev16.hatenablog.com/entry/2019/06/20/010121:embed:cite]



## 参考資料

[1] imageworks, "OpenColorIO-Configs", https://github.com/imageworks/OpenColorIO-Configs

[2] ampas, "ctlrender", https://github.com/ampas/CTL/tree/master/ctlrender

[3] toruのブログ, "CTLで記述された ACES の RRT と ODT を画像に適用する", https://trev16.hatenablog.com/entry/2019/06/01/155644
