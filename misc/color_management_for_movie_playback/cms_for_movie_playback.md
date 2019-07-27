# 動画ファイル再生時のカラーマネジメント

## 目的

* ローカルファイル、各種配信サイト(Youtube, Netflixなど)の動画再生時のカラーマネジメントの挙動を調査する
* ローカルファイル
  * Windows の 標準再生プレイヤー
  * MPC BE
  * VLC
* 配信サイト
  * YouTube
  * Netflix
  * Vimeo
* ブラウザ
  * Google Chrome
  * Firefox
  * Vivaldi
  * Edge
* 動画再生プレイヤー内部の話
  * Splitter
  * Decoder
    * 吐き出す形式って YUV422 Limited なのかしら？
  * Video Renderer
    * YUV2RGB をするの？
    * 解像度の変換もするの？
    * カラマネするのはココ？それとももっと後ろ？

## 背景

2010年代後半になり様々な Color Space の動画ファイルを見かけるようになった。以前は BT.709 しか存在しなかったが、最近では 