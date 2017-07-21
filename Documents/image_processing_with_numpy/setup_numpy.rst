=========================
Numpy のセットアップ
=========================

:著者: Toru Yoshihara
:出版社: Yoshihara's image processing technology laboratory

はじめに
--------
本ドキュメントで扱う信号処理にはNumpy(と関連モジュール)が必須である。
そのセットアップ手順を示す。

おことわり
----------
本来は pyenv-virtualenv などを駆使して Python の環境を分けるべきですが、
そこまで言及するとドキュメントの量が膨大になってしまうので、
本資料では Python の仮想環境については取り扱いません。

Numpyおよび関連ライブラリインストール
---------------------------------------------

Anaconda のインストール
^^^^^^^^^^^^^^^^^^^^^^^^^
`Anaconda`_ のダウンロードページに行き、最新版をインストールするだけ。
これで1部を除き必要なライブラリが入る。

.. _Anaconda: https://www.continuum.io/downloads

OpenCV のインストール
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
OpenCV は Anaconda のインストールだけだと入らないので、別途コマンドを叩く必要がある。

.. code:: bat

    > conda install -c https://conda.anaconda.org/menpo opencv3 

基本的には上記コマンドで上手くいくはずだが失敗することもある。
その場合は、現在の ``Python`` の Version が新しすぎて、OpenCV のライブラリが
存在していない可能性が高い。

`ここのリンク`_ を見れば対応している ``Python`` の Version が分かる(気がする)ので
``pyenv`` とかを上手く活用して乗り切ってほしい。

.. _ここのリンク: https://anaconda.org/menpo/opencv3/files

ライブラリへのパスを通す(任意)
-----------------------------------------------
以後に活用するライブラリへのパスを通しておく。
なお、本ライブラリは某エンジニアが勝手に組んだものであるため動作は保証しない。
自己責任で使用すること。

また、本ライブラリは準備しなくても以降のドキュメント解読に支障はない(はず)。

ライブラリのダウンロード
^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: bat

    > git clone https://github.com/toru-ver4/sip.git
    > cd sip
    > git checkout develop


環境変数の設定
^^^^^^^^^^^^^^
``PYTONPATH`` を設定しないと ``import`` 文が面倒になるので設定しておく。

.. code-block:: bash

    # 本当は Windows 環境を想定してるけど、書き方が分からないので Bash風に書いた
    $ export PYTHONPATH=${PYTHONPATH}:${WORKING_DIRECTORY}sip/lib


``WORKING_DIRECTORY`` は ``git clone`` したディレクトリを意味する。


番外：sphinxのセットアップ
--------------------------------
.. code-block:: bat

    > pip install nbsphinx
    > conda update ipython -c conda-forge
    > conda update ipykernel
    > pip install sphinx-autobuild
