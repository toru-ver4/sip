# OCIO Study

## ゴール

以下2点を理解して OCIO のカスタマイズができるようになる。

* config.ocio の文法理解
* aces_1.0.3 の config.ocio 生成スクリプトの動作を理解する

## config.ocio の理解

構成は以下の通り。

1. Header
2. roles
3. displays
4. colorspaces

### 1. Header

短いので抜粋。

```ocio
ocio_profile_version: 1

search_path: luts
strictparsing: true
luma: [0.2126, 0.7152, 0.0722]

description: An ACES config generated from python
```

```search_path``` のディレクトリは検索するっぽいよ。

```luma``` は何に使うんだろ。以下に書いてあるのはBT.709のY計算式


### 2. roles

短いので抜粋。

```ocio
roles:
  color_picking: Output - Rec.709
  color_timing: ACES - ACEScc
  compositing_linear: ACES - ACEScg
  compositing_log: Input - ADX - ADX10
  data: Utility - Raw
  default: ACES - ACES2065-1
  matte_paint: Utility - sRGB - Texture
  reference: Utility - Raw
  rendering: ACES - ACEScg
  scene_linear: ACES - ACEScg
  texture_paint: ACES - ACEScc
```

カテゴリ毎のデフォルトの colorspace を指定しているっぽい。

なお、colorspace は config.ocio の下の方で定義してある。

### 3. displays

一部省略して抜粋。

```ocio
displays:
  ACES:
    - !<View> {name: sRGB, colorspace: Output - sRGB}
    - !<View> {name: DCDM, colorspace: Output - DCDM}
    - !<View> {name: DCDM P3 gamut clip, colorspace: Output - DCDM (P3 gamut clip)}
    - !<View> {name: P3-D60, colorspace: Output - P3-D60}
    - !<View> {name: P3-D60 ST2084 1000 nits, colorspace: Output - P3-D60 ST2084 (1000 nits)}
    ST2084 (2000 nits)}
    - !<View> {name: P3-D60 ST2084 4000 nits, colorspace: Output - P3-D60 ST2084 (4000 nits)}
    - !<View> {name: P3-DCI, colorspace: Output - P3-DCI}
    - !<View> {name: Rec.2020, colorspace: Output - Rec.2020}
    - !<View> {name: Rec.2020 ST2084 1000 nits, colorspace: Output - Rec.2020 ST2084 (1000 nits)}
    - !<View> {name: Rec.709, colorspace: Output - Rec.709}
    - !<View> {name: Raw, colorspace: Utility - Raw}
    - !<View> {name: Log, colorspace: Input - ADX - ADX10}

active_displays: [ACES]
active_views: [sRGB, DCDM, DCDM P3 gamut clip, P3-D60, P3-D60 ST2084 1000 nits, P3-D60 ST2084 2000 nits, P3-D60 ST2084 4000 nits, P3-DCI, Rec.2020, Rec.2020 ST2084 1000 nits, Rec.709, Rec.709 D60 sim., sRGB D60 sim., Raw, Log]
```

まず前半で ```ACES```というカテゴリのリストを作っている。名前を対応する colorspace を1行ずつ定義している。

後半では（おそらく）デフォルトの display として ```active_displays: [ACES]``` を宣言している。

```active_views:``` は何のために存在してるんだろ。宣言はしたけど有効化しないってのを許可してるのかしら？

### 4. colorspaces

幾つかパターンがある。分けて解析する。

#### 一般的なパターン

S-Log3, S-Gamut3 の colorspace 記述例を以下に抜粋する

```ocio
  - !<ColorSpace>
    name: Input - Sony - S-Log3 - S-Gamut3
    family: Input/Sony
    equalitygroup: ""
    bitdepth: 32f
    description: |
      S-Log3 - S-Gamut3

      ACES Transform ID : IDT.Sony.SLog3_SGamut3_10i.a1.v1
    isdata: false
    allocation: uniform
    allocationvars: [0, 1]
    to_reference: !<GroupTransform>
      children:
        - !<FileTransform> {src: S-Log3_to_linear.spi1d, interpolation: linear}
        - !<MatrixTransform> {matrix: [0.752983, 0.14337, 0.103647, 0, 0.0217077, 1.01532, -0.0370265, 0, -0.00941605, 0.00337042, 1.00605, 0, 0, 0, 0, 1]}
```

```description``` の ```ACES Transform ID``` って何だ？

ACESで使うものであり、OCIOでは必須ではない。省略可能。Canon-Log3 にはタグ無し。

allocation は linear か log かの識別子。詳細は [ここ](http://opencolorio.org/configurations/allocation_vars.html) 。
  * linear: uniform
  * log: lg2

allocationvars は 3DLUT の入力レンジを示す。```.spi3d``` ファイルに記述すればよくね？
とも思ったが、そもそも ```.spi3d``` はヘッダに入力レンジを記載する領域が無かった。
余談だが、```.spi1d``` には記述する領域がある。

上記の ```<MatrixTransform>``` の係数は S-Gamut3 to ACES AP0 変換Matrixだった。
よって入力のWhite(D65)をACESのWhite(D60) に変換している。

その際に **chromatic adaptation** として **CAT02** を使っていることに注意したい。
Bradfordじゃなかった…。

#### ACES特有の(？)パターン

ディスプレイ出力用の Rec.2020 ST2084 (1000 nits) の記述を以下に抜粋する。

```ocio
  - !<ColorSpace>
    name: Output - Rec.2020 ST2084 (1000 nits)
    family: Output
    equalitygroup: ""
    bitdepth: 32f
    description: |
      ACES 1.0 Output - Rec.2020 ST2084 (1000 nits) Output Transform
      ACES Transform ID : ODT.Academy.Rec2020_ST2084_1000nits.a1.0.3
    isdata: false
    allocation: uniform
    allocationvars: [0, 1]
    to_reference: !<GroupTransform>
      children:
        - !<FileTransform> {src: InvRRT.Rec.2020_ST2084__1000_nits_.Dolby_PQ_1000_nits_Shaper.spi3d, interpolation: tetrahedral}
        - !<FileTransform> {src: Dolby_PQ_1000_nits_Shaper_to_linear.spi1d, interpolation: linear}
    from_reference: !<GroupTransform>
      children:
        - !<FileTransform> {src: Dolby_PQ_1000_nits_Shaper_to_linear.spi1d, interpolation: linear, direction: inverse}
        - !<FileTransform> {src: Dolby_PQ_1000_nits_Shaper.RRT.Rec.2020_ST2084__1000_nits_.spi3d, interpolation: tetrahedral}
```
