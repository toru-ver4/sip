# YCbCr変換による色数の減少

## 目的

* RGB --> YCbCr --> RGB の変換で色数がどれだけ減るかを調べる。
* Full Range と Limited Range の双方で比較する。

## 背景

YCbCr変換[1]、および Chroma Subsampling[2] による情報圧縮は大変素晴らしい技術である。
視覚的な情報のロスを抑えた上でデータを大幅に削減することが出来るため、我々は様々な場面でその恩恵を受けている。

その一方で、YCbCr変換によって生じる情報欠損た対する影響範囲を正確に理解することも重要である。
この記事では RGB --> YCbCr --> RGB の変換で具体的にどの程度の情報欠落が生じるかを調査する。

## 結論


## 調査手順

データのbit深度を$n$とする。この時、RGB空間で表現できる色数$N_n$は以下の式で計算できる。

$$
N_n = 2^{3n}
$$

続いて、RGB --> YCbCr --> RGB を行った後の色数を $N_n'$ とする。
今回求める値は $\frac{N_n}{N_n'}$ である。

$N_n'$はオリジナルのRGB値を $(R_i, G_j, B_k)$、YCbCr の逆変換で得られたRGB値を $(R'_i, G'_j, B'_k)$ とすると、

$$
(R_i, G_j, B_k) = (R'_i, G'_j, B'_k)
$$

が成立する個数を意味する。なお、$i, j, k$ は $0, 1, ..., n-2, n-1$ の整数である。

## 考察


## 参考文献

[1] YCbCr, https://en.wikipedia.org/wiki/YCbCr, (2019/03/04確認)

[2] Chroma subsampling, https://en.wikipedia.org/wiki/Chroma_subsampling (2019/03/04確認)
