

1. 安装附件中的vivehub.exe

2. 安装steam与steamvr

3. 找到steam安装路径下的Steam\config\steamvr.vrsettings 文件，加入如下代码：  
"steamvr" : {

"requireHmd" : false,

"activateMultipleDrivers" : true,

"neverOpenServer" : false

}

 

4. 将vive接收器连接至电脑，启动安装好的vivehub

 

进入vive hub设置界面，选择“VIVE自定位追踪器“，首先进行设备配对，配对完成后，点下面的”开始设置“

5. 启动steamvr，界面内有tracker标识则表示连接成功

 

6. 连通后，文件夹中的vive_link_test.py可以读到vive的数据
