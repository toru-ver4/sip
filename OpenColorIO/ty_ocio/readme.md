# 自前の OpenColorIO Config を作成する

## 手順

注意：1回目はエラーが出ます。再帰的な処理が必要です

1. ```make_ocio_lut_and_matrix.py``` を編集して利用する Color Space の LUT と Matrix を作成する。
2. ```make.sh``` を実行する。エラーが出る。
3. 手順2 で吐き出されたログメッセージを使い、```make_ocio_color_space.py``` を編集して使用する Color Space を定義する。
4. ```make_ocio_config.py``` で role や Color Space などを記述する。
5. ```make.sh``` を実行する。ocio.config が正しく生成される…(はず)。

## OpenColorIO について

情報がとっちらかってイマイチだがブログの方に書く予定。
