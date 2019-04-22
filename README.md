# Mac下安卓应用的tensorflow迁移学习

在 [ABO-detector](https://github.com/davelet/ABO-detector) 的基础上，将PC上的图片集训练到安卓上。

参考资料：[How to Retrain an Image Classifier for New Categories](https://www.tensorflow.org/hub/tutorials/image_retraining)

## 安卓环境搭建

1. 下载android studio 并安装：https://jingyan.baidu.com/article/20b68a88ff2ca4796cec6232.html
2. （可选）检验环境：https://developer.android.com/training/basics/firstapp/creating-project
3. 下载： https://github.com/tensorflow/examples 尝试下面“通过安卓观察效果” 的第一步。

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

（会先生成瓶颈文件，根据文件的数据时间难以估计）

训练好以后，会在`/tem/` 目录下生成 `.pb` 文件和 `output_labels.txt` 文件。

## 通过安卓观察效果

### 1. 先克隆tensorflow example库
 https://github.com/tensorflow/examples 。

打开 `lite/examples/image_classification` 目录，用 Android Studio 打开 `Android/app` 目录。Android Studio会自动编译，编译成功后先安装apk到自己手机看一下。

### 2. 生成tflite文件

打开convertor.py文件，编辑它：

- graph_def_file改成上面的pb路径
- input_arrays改成Placeholder
- output_arrays改成final_result

原因见[Abo_detection](https://github.com/davelet/ABO-detector#%E8%BE%93%E5%85%A5%E8%BE%93%E5%87%BA%E5%BC%A0%E9%87%8F%E8%AE%B0%E5%BD%95)。

执行convertor.py很快就生成converted_model.tflite文件。

### 3. 替换安卓资源

把生成的tflite文件和第一步的output_labels.txt文件一起复制到 `image_classification/android/app/src/main/assets/` 下面。

找到 `ClassifierFloatMobileNet.java` 文件，复制一份并改名为 `ClassifierFloatInception.java`。修改其中image size XY 都是299（原因见[InceptionV4](/InceptionV4.pdf)）。修改modelPath和LabelPath为上面刚复制过去的文件。

修改`ClassifierActivity.java` 中的`classifier`变量初始化为`new ClassifierFloatInception(this)`。

### 4. 重新build
再次打包安装到手机上可以用手机识别不同花朵的图片试一下。
 
只能识别菊花、蒲公英、向日葵、郁金香、玫瑰。

## 训练kaggle集

图片分类和对象识别不同的是，对象识别需要标记；而图片分类会把整张图片当成某一个东西。

把[Abo_detection](https://github.com/davelet/ABO-detector#%E8%BE%93%E5%85%A5%E8%BE%93%E5%87%BA%E5%BC%A0%E9%87%8F%E8%AE%B0%E5%BD%95)中的水果图片集删掉xml标记数据和mixed多对象文件，把剩下的分成三类分别放进apple/banana/orange目录。重新训练。

因为图片太少，训练会很快结束并且标明识别率高达100%。

识别效果：[视频](/水果识别.mp4)


