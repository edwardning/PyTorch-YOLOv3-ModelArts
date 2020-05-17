# PyTorch-YOLOv3-ModelArts
在华为云ModelArts平台部署PyTorch版本的YOLOv3目标检测网络，实现模型训练、在线预测及参赛发布。


- 动机

    正在参加“华为云杯”2020深圳开放数据应用创新大赛·生活垃圾图片分类比赛，官方只提供了keras版本YOLOv3的baseline。
但该baseline判分只有0.05分，低的可怕，远远达不到YOLOv3应有的水平。

- What I do

    自己keras用的比较少，因此没去深究官方baseline哪里出了问题。
索性自己写了个PyTorch版本的baseline。经测试，性能大幅大幅大幅提升。。。（看结果请移步最后）
果真官方baseline有问题，有兴趣的小伙伴可以考究一下。


- source code: https://github.com/eriklindernoren/PyTorch-YOLOv3
- 大赛地址: https://competition.huaweicloud.com/information/1000038439/introduction

## 使用前准备
##### 解压官方原始数据集，制作新数据集
    $ cd PyTorch-YOLOv3-ModelArts/my_utils
    $ python prepare_datasets.py --source_datasets --new_datasets

##### 下载预训练模型
    $ cd weights/
    $ bash download_weights.sh

##### 创建自定义模型的cfg文件
    $ cd PyTorch-YOLOv3-ModelArts/config
    $ bash create_custom_model.sh <num-classes> #此处已创建，即yolov3-44.cfg
    
## 在ModelArts平台上训练
1.将新数据集打包成压缩文件，替换原始数据集压缩包；

2.训练集和测试集的图片路径默认保存在config/train.txt和valid.txt中，每一行代表一张图片，默认按8：2划分。注意每行图片的路径为虚拟容器中的地址，自己重新划分训练集时只需要修改最后的图片名称，千万不要更改路径！

2.如果使用预训练模型，请提前将其上传到自己的OBS桶中，并添加参数

`--pretrained_weights = s3://your_bucket/{model}`。

此处的model可以是官方预训练模型（yolov3.weights或darknet53.conv.74），也可以是自己训练过的PyTorch模型（.pth）。

3.训练过程中，学习率等参数默认不进行调整，请依个人经验调整

4.其余流程同大赛指导文档。

## 测试
1. 与官方keras版本的baseline比较，训练速度提升两倍多（官方baseline跑10个epoch需要150分钟，本项目仅需47分钟）；参赛发布大概一小时完成判分，同样快一倍以上。

2. 官方baseline跑10个epoch用时两个半小时，判分却仅得0.05；本项目只训练头部跑5个epoch仅仅用时17分钟，判分达到0.17（惊掉下巴）
`原PyTorch版的代码中采用了gradient_accumulations，默认为2，即每两个iteration更新一次梯度，因此训练速度会大幅加快。`
`这样做会带来精度上的损失，为提高mAP可将此值设为1。不过经测试，速度仍快于官方keras版本baseline`

3. 因为比赛刚开始，过多的测试就不做了。个人估计，在此baseline上改进，最终成绩可以达到0.6分左右。
当然，如果想拿奖金的话还是转投RCNN或者EfficientDet吧。


## Credit

### YOLOv3: An Incremental Improvement
_Joseph Redmon, Ali Farhadi_ <br>

**Abstract** <br>
We present some updates to YOLO! We made a bunch
of little design changes to make it better. We also trained
this new network that’s pretty swell. It’s a little bigger than
last time but more accurate. It’s still fast though, don’t
worry. At 320 × 320 YOLOv3 runs in 22 ms at 28.2 mAP,
as accurate as SSD but three times faster. When we look
at the old .5 IOU mAP detection metric YOLOv3 is quite
good. It achieves 57.9 AP50 in 51 ms on a Titan X, compared
to 57.5 AP50 in 198 ms by RetinaNet, similar performance
but 3.8× faster. As always, all the code is online at
https://pjreddie.com/yolo/.

[[Paper]](https://pjreddie.com/media/files/papers/YOLOv3.pdf) [[Project Webpage]](https://pjreddie.com/darknet/yolo/) [[Authors' Implementation]](https://github.com/pjreddie/darknet)

```
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
```
