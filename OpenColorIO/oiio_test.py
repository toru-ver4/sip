import OpenImageIO as oiio
from OpenImageIO import ImageInput, ImageOutput
from OpenImageIO import ImageBuf, ImageSpec, ImageBufAlgo

if __name__ == '__main__':
    print(dir(oiio))
    # t = oiio.TypeDesc(oiio.UINT8)
    # print(t)
    # print(oiio.TypeDesc.TypeInt)
    input = oiio.ImageInput.open("test.png")
    print(input.format_name())
    spec = input.spec()
    print(spec.width, spec.height)
    input.close()
    print(dir(ImageBufAlgo.colorconvert))

