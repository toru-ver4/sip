# Encode

## 一般的なオプションを指定してYUV420作る

```
ffmpeg -r 60 -i img/file_%08d.dpx -r 60 -vf scale=in_range=full:out_range=tv -c:v libx265 -x265-params crf=16:colorprim=9:transfer=16:max-cll=1000,400:master-display=G(8500,39850)B(6550,2300)R(35400,14600)WP(15635,16451)L(10000000,10) -pix_fmt yuv420p10le normal.mp4
```

## 入力をLimitedだとちょろまかしてYUV420を作る

```
ffmpeg -r 60 -i img/file_%08d.dpx -r 60 -vf scale=in_range=tv:out_range=tv -c:v libx265 -x265-params crf=16:colorprim=9:transfer=16:max-cll=1000,400:master-display=G(8500,39850)B(6550,2300)R(35400,14600)WP(15635,16451)L(10000000,10) -pix_fmt yuv420p10le normal_like_full.mp4
```

## 時代は ProRes

```
ffmpeg -r 60 -i img/file_%08d.dpx -r 60 -vf scale=in_range=tv:out_range=tv -c:v prores_ks -profile 3 -pix_fmt yuv422p10le normal_like_full.mov
ffmpeg -r 60 -i img/file_%08d.dpx -r 60 -vf scale=in_range=full:out_range=tv -c:v prores_ks -profile 3 -pix_fmt yuv422p10le normal.mov
```