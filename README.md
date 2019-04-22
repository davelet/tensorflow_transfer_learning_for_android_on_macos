# Mac下安卓应用的tensorflow迁移学习

在 [ABO-detector](https://github.com/davelet/ABO-detector) 的基础上，将PC上的图片集训练到安卓上。

参考资料：[How to Retrain an Image Classifier for New Categories](https://www.tensorflow.org/hub/tutorials/image_retraining)

## 使用inception v3识别花朵
首先克隆tensorflow-hub库 https://github.com/tensorflow/hub 。
找到 `tensorflow_hub/pip_package/setup.py` 所在的目录，执行
```
python setup.py build

python setup.py install
```

然后下载花朵图片集（目录可自行选择）：
```
cd ~

curl -LO http://download.tensorflow.org/example_images/flower_photos.tgz

tar xzf flower_photos.tgz
```
找到 `examples/image_retraining/retrain.py` 所在的目录，执行

```
python retrain.py --image_dir ~/flower_photos
```
对图片的训练就开始了，这个要花一段时间。默认训练4000步。

训练好以后，会在`/tem/` 目录下生成 `.pb` 文件和 `output_labels.txt` 文件。

## 通过安卓观察效果

