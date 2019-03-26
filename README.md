# 使用文档

#### 一、训练
1. 修改lstm_ctc/config/config_demo.yaml, 按照提示修改以下内容:  
(1) 训练集和测试集目录（TrainSetPath、TestSetPath）  
(2) 改图片标签的正则表达式（LabelRegex）   
(3) 修改最大标签长度（MaxTextLength）   
(4) 修改图片的高和宽（IMG_H, IMG_W）   
(5) 设置生成的模型的名称（ModelName）   
(6) 设置标签中包含的所有字符(Alphabet)    

2. 训练是根据需要调整NeuralNet下的配置：  
(1) RNNSize:默认64, 或者128    
(2) Dropout:默认0.25, 或者0.5    

3. 根据需要调整训练参数TrainParam：   
(1) 一般情况下Epochs=2~5即可训练处较高的准确率   
(2) BatchSize的值默认128, 或者32, 64等   
(3) TestBatchSize: 一般用默认值即可    
(4) TestSetNum: 设置测试集数量, 默认1000

4. python keras_image_ocr_v1.py启动训练  
(1) keras_image_ocr_v1.py是将图片转为灰度图进行训练   
(2) keras_image_ocr_v2.py是将图片直接使用RGB通道图片进行训练    

5. 生成的模型在lstm_ctc/model目录下

#### 二、web测试
1. 将模型文件拷贝到web/captcha_ocr/model目录下     
2. 修改run.py和utils.py文件
3. python run.py启动项目进行测试
