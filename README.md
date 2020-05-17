# PyTorch-YOLOv3-ModelArts
在华为云ModelArts平台部署PyTorch版本的YOLOv3并实现训练、在线预测及参赛发布。

正在参加“华为云杯”2020深圳开放数据应用创新大赛·生活垃圾图片分类比赛，官方只提供了keras版本的baseline。
自己keras用的比较少，因此写了个PyTorch版本的baseline。

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

`--pretrained_weights = s3://your_bucket/darknet53.conv.74 (or .pth)`。

注意应使用darknet53.conv.74而不是yolov3.weights。

3.训练过程中，学习率等参数默认不进行调整，请依个人经验调整

4.其余流程同大赛指导文档。

#### 测试
1.与官方keras版本的baseline比较，训练速度快一倍以上（单个epoch需4-5分钟）；参赛发布仅需50分钟完成判分，同样快一倍以上。

`原PyTorch版的代码中采用了gradient_accumulations，默认为2，即每两个iteration更新一次梯度，因此训练速度会大幅加快。`
`这样做会带来精度上的损失，为提高mAP可将此值设为1。不过经测试，速度仍快于官方keras版本baseline`

2.默认不采用两阶段训练，并且学习率保持不变，因此mAP会比baseline低。稍微改进后会大幅提升。


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
