REM ffmpeg -framerate 60 -i src_img_%%4d.png -c:v h264 -crf 18 -colorspace bt709 -color_trc bt709 -color_primaries bt709 -vf setparams=field_mode=prog:range=tv:color_primaries=bt709:color_trc=bt709:colorspace=bt709 -pix_fmt yuv420p codec_709_filter_709.mp4 -y
REM ffmpeg -framerate 60 -i src_img_%%4d.png -c:v h264 -crf 18 -colorspace bt709 -color_trc bt709 -color_primaries bt709 -pix_fmt yuv420p codec_709_filter_none.mp4 -y

REM ffmpeg -framerate 60 -i src_img_%%4d.png -c:v h264 -crf 18 -colorspace bt2020nc -color_trc bt709 -color_primaries bt2020 -vf setparams=field_mode=prog:range=tv:color_primaries=bt2020:color_trc=bt709:colorspace=bt2020nc -pix_fmt yuv420p codec_2020_filter_2020.mp4 -y
REM ffmpeg -framerate 60 -i src_img_%%4d.png -c:v h264 -crf 18 -colorspace bt2020nc -color_trc bt709 -color_primaries bt2020 -pix_fmt yuv420p codec_2020_filter_none.mp4 -y

REM ffmpeg -loop 1 -i "ColorChecker_All_ITU-R BT.2020_D65_BT1886_Reverse.tiff" -r 60 -t 3 -c:v h264 -crf 18 -colorspace bt2020nc -color_trc bt709 -color_primaries bt2020 -vf setparams=color_primaries=bt2020:color_trc=bt709:colorspace=bt2020nc -pix_fmt yuv420p color_checker_bt2020.mp4 -y

ffmpeg -loop 1 -i "SMPTE ST2084_ITU-R BT.2020_D65_1920x1080_rev01_type1.tiff" -r 24 -t 3 -c:v libaom-av1 -strict -2 -crf 18 -colorspace bt2020nc -color_trc smpte2084 -color_primaries bt2020 -vf setparams=color_primaries=bt2020:color_trc=smpte2084:colorspace=bt2020nc -pix_fmt yuv420p10le tp_st2084_bt2020.mp4 -y
