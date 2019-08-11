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

### SEI

* NAL の1種。非VCL-NALユニットに属する。

### VUI

* NAL の1種である SPS に含まれる情報の１つ。

### 全体構成

https://yumichan.net/video-processing/video-compression/introduction-to-h264-2-sodb-vs-rbsp-vs-ebsp/

https://github.com/uupaa/H264.js/wiki/EBSP,-RBSP,-SODB

RBSP(raw byte sequence payload)

### ffmpeg で h265 の ビットストリームを作る方法

```
ffmpeg -loop 1 -i "SMPTE ST2084_ITU-R BT.2020_D65_1920x1080_rev03_type1.dpx" -r 24 -t 3 -f yuv4mpegpipe -strict -1 -pix_fmt yuv422p10le tp_st2084_bt2020.y4m -y
x265-64bit-10bit-2019-07-29 tp_st2084_bt2020.y4m tp_st2084_bt2020.bin
```
