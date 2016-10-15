# install

1. Visual Studio Community 2015 をインストール
2. CUDA Toolkit をインストール
3. PyCuda を pip でインストール
4. Path に以下を追加

```
C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin;C:\Program Files (x86)\Microsoft Visual Studio 14.0\Common7\IDE;C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\amd64
```

5.  `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin\nvcc.profile` に
Visual Studio の include を追加。

```
INCLUDES        +=  "-I$(TOP)/include" "-IC:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\include" "-IC:\Program Files (x86)\Windows Kits\10\Include\10.0.10240.0\ucrt" $(_SPACE_)
```