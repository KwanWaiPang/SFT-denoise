from PIL import Image, ImageDraw
import glob, os
 
dir = "/media/sdc/wpguan/zhangxinjie/paper_sft_denoise/result/results/test_basicmodelx4_c60/RN15/"  # the direction of the result
 
# the(x, y), and the width
x = 160
y = 80
width = 100
scale=2
 
# capture an image
pyFile = glob.glob(os.path.join(dir, "*.png"))
pyFile += glob.glob(os.path.join(dir, "*.jpg"))
pyFile += glob.glob(os.path.join(dir, "*.bmp"))
result_path = os.path.join(dir,"result")
 
# if the in result
if not os.path.exists(result_path) :
        os.mkdir(result_path)

print('~~~~~the number of datas is: ',len(pyFile))
# Traverse the picture
for img_path in pyFile:
    im = Image.open(img_path)
    draw = ImageDraw.Draw(im)
 
    aspect_ratio = 1#im.size[0]/im.size[1] # Aspect ratio
    # Intercepting a selection image
    im_ = im.crop((x, y, x+width, (y+width)//aspect_ratio))
    # Box out of the selection
    draw.rectangle((x, y, x+width, (y+width)//aspect_ratio), outline='red', width=3) # width is the width of the line
 
    #im_ = im_.resize(im.size) # Call the resize function to enlarge the submap to the original image size
    width1=int(im_.size[0]*scale)
    height1=int(im_.size[1]*scale)
    im_=im_.resize((width1, height1), Image.ANTIALIAS)
 
    # Get the file name
    _, img_name = os.path.split(img_path)
    img_name, _ = os.path.splitext(img_name)
    print(img_name)
 
    # Save submap and original image with marquee
    im_.save(os.path.join(result_path , img_name + '_sub_image.png'))
    im.save(os.path.join(result_path , img_name + '_ori_image.png'))
