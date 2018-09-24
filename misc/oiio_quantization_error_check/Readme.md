# OIIO の量子化誤差調査

## 背景

oiio を使って 10bit or 12bit の dpx ファイルを保存したい。しかし、oiio の I/F だと uint16 でデータをやりとりする必要がある。
oiio の内部量子化？などによって誤差が生じないか大変不安である。問題が無いか確認を行う。

## 手順

1. 10bit or 12bit の Rampパターンを作成
2. [0.0:1.0]に正規化してDouble型に
3. oiio で save and close.
4. oiio で open and read.
5. 整数型に正規化して10bit/12bitに。
6. 最初の値からズレていないか verifyする。

## 結論

大丈夫でした。詳細は ```test_10bit_error(), test_12bit_error``` を参照。
