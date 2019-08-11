# Dolby Cinema 向けの Output Transform を少しだけ調査した

## 目的

* Dolby Cinema に使われている Output Transform の特性を軽く調べる
  * 具体的には Linear なグレースケールを入力した時の出力特性を調べる。

## 背景

HOTSHOT の [Dolby Cinema とは何か？](https://hotshot-japan.com/feature/11-dolbycinema-jp02-2/) を読んでいたところ、以下の記述が目に留まった[1]。

> 今回ハリウッドにおけるドルビーシネマへの変換作業に備えて、事前に日本では1000nitsのモニターで調整したHDRデータを持って行きましたが、実際にドルビーシネマスクリーンの4Kデュアルプロジェクションで見る108nitsのHDR映像では、モニターとはその見え方が大きく異なり、結果的に現地で全てデータを作り直すことになりました。

筆者は以前より、業務用モニター向けの 1000 nits 出力と Cinema 向けの 108 nits 出力との差異が気になっており、機会があれば調査したいと考えていた。

最近、ACES 関連の勉強をしていたところ、ACES V1.1 にて Dolby Cinema 向けの Output Transform が追加されていることに気づいた[2]。ちょうど良い機会だと考えて簡単な調査を行った。

※Output Transform などの基本的な情報は [Filmlightさんの資料](http://filmlight.jp/files/20170907.pdf) が分かりやすいです[3]

## 結論

グレースケールに対して以下2つの OutputTransform を適用した結果を図1に示す。

* RRTODT.Academy.P3D65_108nits_7.2nits_ST2084.ctl
* RRTODT.Academy.Rec2020_1000nits_15nits_ST2084.ctl

![zu](./comparison_108_vs_1000.png)

## やり方

`ctlrender` コマンドを使って .ctl をグレースケールの .exr ファイルに適用するだけ。

`ctlrender` の使い方は[この記事](https://trev16.hatenablog.com/entry/2019/06/01/155644) を参照[4]。

## 考察

1000nits の OutputTransform を使って作り込んだコンテンツに 108nits の OutputTransform を当てると、ハイライトの見え方が相当に変わると考える。1000nits の業務用モニターでは表現可能だった 500～1000nits の領域は、**何の追加処理もしなかった場合** 108nits の OutputTransform を適用すると殆ど潰れてしまう。

現場ではどういう運用を想定しているのだろうか。大変興味がある。

## 参考資料

[1] HOTSHOT, "Dolby Cinemaとは何か？", https://hotshot-japan.com/feature/11-dolbycinema-jp02-2/

[2] ampas, "ampas/aces-dev at v1.1", https://github.com/ampas/aces-dev/tree/v1.1

[3] 松井 幸一, "もう一度ACES",  http://filmlight.jp/files/20170907.pdf

[4] toruのブログ, "CTLで記述された ACES の RRT と ODT を画像に適用する", https://trev16.hatenablog.com/entry/2019/06/01/155644
