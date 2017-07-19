====================
色変換
====================

:著者: Toru Yoshihara (石川県の上位DDRer。最近はHDR-DDR動画の撮影に夢中)
:出版社: Laala Publication

はじめに
--------
さまざまな色空間が混在する現在では色の変換処理がどうしても必要になる。
それを Numpy を使って簡単に実現する。

本ドキュメントでは最初に色変換の簡単な例を示す。その後で
色変換に使う信号処理について解説する。

色域変換
^^^^^^^^^^^^^^^^^^^^^

白色点変換
^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    import matplotlib.pyplot as plt
    import color_convert as cc
    import numpy as np
    import cv2
    %matplotlib inline
    
    # 画像ファイル読み込み
    # --------------------------
    img_file_name = "./figures/source.tif"
    before_img = cv2.imread(img_file_name, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    before_img = before_img[:, :, ::-1]
    
    # 変換のために 1.0 に正規化
    # --------------------------
    img_max_value = np.iinfo(before_img.dtype).max
    after_img = before_img / img_max_value
    
    # ガンマ解除してリニア空間に戻す
    # --------------------------------
    after_img = after_img ** 2.2
    
    # D65 --> D50 に変換
    # -------------------------
    src = cc.const_d65_large_xyz
    dst = cc.const_d50_large_xyz
    convert_matrix = cc.get_white_point_conv_matrix(src, dst)
    after_img = cc.color_cvt(after_img, convert_matrix)
    
    # オーバーフロー、アンダーフローの処理
    # --------------------------------------
    ng_idx_overflow = after_img > 1.0
    ng_idx_underflow = after_img < 0.0
    after_img[ng_idx_overflow] = 1.0
    after_img[ng_idx_underflow] = 0.0
    
    # 1.0 に正規化していたのを元に戻す
    # ---------------------------------
    after_img = after_img ** (1 / 2.2)
    after_img = np.round(after_img * img_max_value).astype(before_img.dtype)
    cv2.imwrite("hoge.tif", after_img[:, :, ::-1])
    
    # 比較表示
    # ---------------------------------
    fig = plt.figure(figsize=(32, 18))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.set_axis_off()
    ax2.set_axis_off()
    ax1.imshow(before_img / img_max_value)  # 整数型だと乱れたので浮動小数点型にした
    ax2.imshow(after_img / img_max_value)
    plt.show()



.. image:: figures/output_1_0.png

