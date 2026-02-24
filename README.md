# pix2pix-translate-line-to-image

1. clone the project from here to your mechine
code:
git clone https://github.com/Guiarguy/pix2pix-translate-line-to-image.git

# maybe you will stuck here then you need to repair and use lfs use pull remain file down
then you need to ctrl+c stop it and repair it
code for repair:
git restore --source=HEAD :/

code for pull it again:
git lfs pull

2. using docker to run this project, so you need to install docker desktop first
   Then you can start build the dockerfile and run it

code for build image:
docker build -t pix2pix-line2img .
#note that the name of image is pix2pix-line2img

code for run:
docker run --rm -it -p 7861:7861 pix2pix-line2img
#note that I use 7861 as port and the container will automatically remove after terminate
