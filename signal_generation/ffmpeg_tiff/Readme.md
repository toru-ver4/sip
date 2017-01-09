# タイトル
Windows PC で HDR動画を YouTube にアップロードする方法

# この記事の目的
Windows PC で HDR動画を YouTube に上げる際、
[公式のドキュメント](https://support.google.com/youtube/answer/7126552?hl=ja)通りに実行しても上手く行かなかったので情報を残しておく。

前提条件として以下の2点はクリアしているものとする。

1. YouTubeヘルプの [アップロードする動画の推奨エンコード設定](https://support.google.com/youtube/answer/1722171) を読んで理解できること。
2. YouTubeヘルプの [HDR 動画をアップロードする](https://support.google.com/youtube/answer/7126552?hl=ja) を読んで理解できること。

# 結論
以下の2点に注意すれば良い。

## 1. Davinci Resolve でエクスポートしたデータを直接アップロードしない
Winodw版の DaVinci Resolve では YouTube が指定する HDR動画の要件を満たせない

## 2. エンコード＆メタデータ付与は外部ツールを使って手動で行う
ffmpeg を使い ProRes形式でエンコードを行った後、
[YouTube HDR メタデータ ツール](https://github.com/youtubehdr/hdr_metadata) を使ってメタデータを付与する。

各ツールでの設定は以下の通り。

### ffmpeg の設定

```
ffmpeg.exe -r 24 -i movie.mov -i bgm.wav \
    -ar 48000 -ac 2 -c:a aac -b:a 384k \
    -r 24 -vcodec prores_ks -profile:v 3 -pix_fmt yuv422p10le \
    -b:v 50000k -shortest -y input.mkv
```
※フレームレート設定は動画に合わせて適切なものを設定すること。<br>
※上記設定は動画と音声が別ファイルの場合の例である。もともと音声が含まれる場合は
```-i bgm.wav``` および ```-shortest``` は削除すること。<br>
※ちなみに ```-an``` オプションを使って音声を無しにしても問題は無い。

### YouTube HDR メタデータ ツール の設定

```
mkvmerge.exe \
     -o output.mkv\
     --colour-matrix 0:9 \
     --colour-range 0:1 \
     --colour-transfer-characteristics 0:16 \
     --colour-primaries 0:9 \
     --max-content-light 0:1000 \
     --max-frame-light 0:300 \
     --max-luminance 0:1000 \
     --min-luminance 0:0.01 \
     --chromaticity-coordinates 0:0.68,0.32,0.265,0.690,0.15,0.06 \
     --white-colour-coordinates 0:0.3127,0.3290 \
     input.mkv
```

最後に、できあがった output.mkv を YouTube にアップロードすれば良い。

# 詳細
上記の結論に至った経緯を記しておく。

## 1. Davinci Resolve でエクスポートしたデータを直接アップロードしない
Windows版 の Davinci Resolve で H.264 エンコードをすると 10bit ではなく 8bit で
吐き出される。これは YouTube が指定する HDR動画の要件を満たさない。
Davinci Resolve で ProRes形式が選択できれば問題ないのだが Windows版では非サポートとなっている。

それから DNxHR HQX 形式も試してみたが NG だった。理由は不明。

## 2. エンコード＆メタデータ付与は外部ツールを使って手動で行う
YouTube に HDR動画だと認識させるためには、10bitの動画としてエンコードする必要がある。
一番簡単なのは ProRes 形式でエンコードすることなので、ffmpeg を使ってエンコードした。

また、同時にメタデータも手動で付与する必要がある。
これは公式ドキュメント通りにツールを実行すれば問題なかった。

# 終わりに
やはり動画編集に Windows PC は向いていない気がする。なんか、色々と躓くことが多い。

