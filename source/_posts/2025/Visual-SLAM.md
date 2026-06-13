---
title: 本科毕业设计论文：巷道移动机器人SLAM技术研究
cover: /img/ScreenShot_2026-03-05_190242_417.png
date: 2025-02-6 20:41:20
categories: 
- [文档&笔记]
tags:
- ROS
---
# 计划完成的任务目标
+ 制作一台移动机器人
+ 使用三维激光雷达和毫米波雷达进行建图
+ 使用深度相机和imu数据融合进行建图和导航

# 三维激光雷达建图的实现（LeGO-LOAM）
+ 算法：[LeGO-LOAM](https://github.com/RobustFieldAutonomyLab/LeGO-LOAM)
+ 激光雷达：[RS-Helios-5515](RS-Helios-5515产品手册.pdf)

## 通讯连接和基本配置
首先用交换机连接激光雷达和电脑，为了方便我决定在vmware中开发，系统为`Ubuntu 18.04`安装了`ROS1-melodic`。网络设置为桥接模式，系统内设置网络手动ip`192.168.1.102`，子网掩码`255.255.255.0`（雷达默认ip`192.168.1.200`发送udp数据包到`192.168.1.102`的端口`6699``7788`）

新建ros工作空间后`src`文件夹放入[雷达ros驱动](5515_SDK.rar)和[雷达话题定义(驱动文档也有说明)](rslidar_msg-master.zip)，按照里面的要求进行配置修改即可，以及`LeGO-LOAM`代码包，除了按照文档说明进行修改以外，还要将`CMakeLists.txt`第四行的c++版本14:`set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3")`。

设置雷达数据为我们使用的32线雷达，修改`LeGO-LOAM/include/utility.h`，这里我补充一些注释
```
// Mine
// 传感器垂直方向上的扫描线数量
extern const int N_SCAN = 32;
// 水平方向上扫描一周360度所采集的点数
extern const int Horizon_SCAN = 1800;
// 水平方向上的角度分辨率
extern const float ang_res_x = 0.2;
// 垂直方向上的角度分辨率（某些垂直角度分辨率可能不是均匀的，可以写平均值）
extern const float ang_res_y = 2.2;
// 最底部扫描线的垂直角度
extern const float ang_bottom = 55.0;
// 地面点分割或移除算法中，指定了一个扫描线的索引号。索引号小于或等于groundScanInd的扫描线上的点，会被认为是潜在的地面点
extern const int groundScanInd = 7;
```

由于雷达数据的格式不同，使用[rs_to_velodyne](https://github.com/HViktorTsoi/rs_to_velodyne)进行格式转化，`A ros tool for converting Robosense pointcloud to Velodyne pointcloud format, which can be directly used for downstream algorithm, such as LOAM, LEGO-LOAM, LIO-SAM, etc.`
使用`rostopic list`可以看到代码接收雷达topic`/rslidar_points`转换后发送到`/velodyne_points`
```
triority@ubuntu:~/Desktop/lidar$ rostopic list
/rosout
/rosout_agg
/rslidar_points
/velodyne_points
```

这里我由于openGL驱动问题导致rviz报错闪退，即使不闪退也不显示任何点云，要将其改为软件驱动。由于rviz错误log为空文件导致我很久没有找到问题所在。没有在物理机上试过不知道和vmware有没有关系
```
export LIBGL_ALWAYS_SOFTWARE=1
```

按照LeGO-LOAM文档要求在实际运行和使用rosbag时需要使用不同的roslaunch配置，具体是修改`LeGO-LOAM/launch/run.launch`的第四行，仿真下设置为false否则为true
```
<param name="/use_sim_time" value="false" />
```

由于雷达驱动和LeGO-LOAM都会启动rviz，我这里直接修改雷达驱动启动文件`rslidar_sdk-dev_opt/launch/start.launch`注释掉rviz的启动内容
```
<launch>
  <node pkg="rslidar_sdk" name="rslidar_sdk_node" type="rslidar_sdk_node" output="screen">
    <param name="config_path" value=""/>
  </node>
  <!-- rviz -->
  <!--<node pkg="rviz" name="rviz" type="rviz" args="-d $(find rslidar_sdk)/rviz/rviz.rviz" />-->
</launch>
```

## 建图测试
习惯上我喜欢先单独启动roscore
```
roscore
```

启动雷达：
```
roslaunch rslidar_sdk start.launch
```

启动雷达数据转换：
```
rosrun rs_to_velodyne rs_to_velodyne XYZIRT XIRI
```

启动建图：
```
roslaunch lego_loam run.launch
```

这时就可以使用激光雷达的数据进行建图了，下图是我所在的实验室
![](69a805494e5c171e8fcc88a78ff7393.png)

## 地图保存和数据记录
如果需要保存建好的地图，在建图快要结束时，打开终端订阅全局点云地图的话题`/laser_cloud_surround`保存为bag包，然后转bag为pcd，使用
`pcl_viewer`打开点云地图
```
triority@ubuntu:~/Desktop/lidar$ rosbag record /laser_cloud_surround
[ INFO] [1744615722.083300899]: Subscribing to /laser_cloud_surround
[ INFO] [1744615722.089056296]: Recording to '2025-04-14-00-28-42.bag'.

triority@ubuntu:~/Desktop/lidar$ rosrun pcl_ros bag_to_pcd 2025-04-14-00-28-42.bag /laser_cloud_surround pcd1
Creating directory pcd1
Saving recorded sensor_msgs::PointCloud2 messages on topic /laser_cloud_surround to pcd1
Got 1681 data points in frame /camera_init on topic /laser_cloud_surround with the following fields: x y z intensity
Data saved to pcd1/1744615726.458191872.pcd

triority@ubuntu:~/Desktop/lidar$ pcl_viewer pcd1/1744615726.458191872.pcd
```

此外也可以直接记录雷达原始数据，便于后续对算法进行改进测试比较等

# LeGO-LOAM源码分析

<details>
  <summary>引用一下来自google ai studio的源码分析作为大致方向上的参考</summary>
  好的，我们来大致分析一下 LeGO-LOAM 的源代码结构和核心思想。LeGO-LOAM (Lightweight and Ground-Optimized Lidar Odometry and Mapping) 是对经典 LOAM 算法的改进，特别针对地面车辆（如自动驾驶汽车）的应用场景进行了优化。

**核心目标：** 实现低计算量、实时的激光雷达里程计和建图，并特别优化地面点的处理。

**主要代码结构 (基于常见的 ROS 实现):**

LeGO-LOAM 通常由以下几个核心的 ROS 节点（或 Nodelet）组成，它们通过 ROS Topic 进行通信：

1.  **`imageProjection` 节点 (预处理模块):**
    *   **输入:** 原始激光雷达点云数据 (`sensor_msgs/PointCloud2`)，通常来自 `/points_raw` 或类似 topic。
    *   **核心功能:**
        *   **点云分割 (Segmentation):**
            *   **地面分割 (Ground Segmentation):** 这是 LeGO-LOAM 的关键优化之一。它将点云投影到一个深度图（Range Image）上，然后利用点与点之间的角度关系或简单的平面拟合方法，快速地将地面点与非地面点分离。这极大地减少了后续处理的点云数量，并移除了可能不稳定的地面特征。
            *   **点云聚类 (Point Cloud Clustering):** 对分割出的非地面点进行聚类（例如使用基于距离或连通性的方法），将点云分割成独立的物体或障碍物。小的、稀疏的聚类会被滤除，以消除噪声和不可靠的特征点。
        *   **点云信息标记:** 为每个点打上标签（地面点、分割后的聚类ID等），并计算每个点的距离、角度等信息，方便后续处理。
    *   **输出:**
        *   分割后的点云 (`/segmented_cloud`)：包含非地面点的聚类结果。
        *   地面点云 (`/ground_cloud`)：单独发布的地面点。
        *   外点/噪声点云 (`/outlier_cloud`)。
        *   点云信息 (`/cloud_info`)：包含每个点的标签、范围、起始/结束索引等辅助信息。

2.  **`featureAssociation` 节点 (特征提取与里程计模块):**
    *   **输入:** `imageProjection` 节点输出的分割点云、地面点云和点云信息。
    *   **核心功能:**
        *   **特征提取 (Feature Extraction):**
            *   基于分割后的非地面点云，计算每个点的曲率（Curvature）。
            *   根据曲率大小，选取**边缘点 (Edge Points)** (曲率大的点) 和**平面点 (Planar Points)** (曲率小的点)。
            *   为了提高鲁棒性，会排除一些不稳定的特征点（例如，平行于激光束方向的点、被遮挡的点）。
            *   LeGO-LOAM 的一个特点是它会分别处理边缘特征和平面特征。
        *   **激光雷达里程计 (Lidar Odometry):**
            *   执行**扫描到扫描 (Scan-to-Scan)** 的匹配。
            *   将当前帧提取的边缘点与上一帧的边缘线进行匹配（点到线距离最小化）。
            *   将当前帧提取的平面点与上一帧的平面片进行匹配（点到面距离最小化）。
            *   **两步优化 (Two-Step Optimization):** LeGO-LOAM 的另一个关键优化。它首先利用地面点（被视为一个大平面）和分割出的平面点进行匹配，估计车辆的 [z, roll, pitch]；然后利用边缘点匹配，估计车辆的 [x, y, yaw]。这种分离优化降低了计算复杂度，并提高了对特定运动的估计精度。
            *   使用优化算法（通常是 Levenberg-Marquardt, LM）迭代求解最小化点到线/点到面距离的位姿变换（旋转和平移）。
    *   **输出:**
        *   高频、但可能存在漂移的激光雷达里程计位姿 (`/laser_odom_to_init`)。
        *   当前帧提取的特征点云（角点 `/laser_cloud_corner_last`，平面点 `/laser_cloud_surf_last`），供建图模块使用。

3.  **`mapOptmization` 节点 (建图与位姿优化模块):**
    *   **输入:** `featureAssociation` 节点输出的里程计位姿、特征点云。
    *   **核心功能:**
        *   **地图维护 (Map Maintenance):**
            *   接收来自 `featureAssociation` 的特征点云和粗略位姿。
            *   根据里程计位姿，将历史关键帧的特征点（边缘点和平面点）变换到世界坐标系下，构建一个全局的特征地图（通常用 Voxel Grid 或 K-D Tree 存储，以便快速查找）。
        *   **扫描到地图 (Scan-to-Map) 匹配:**
            *   将当前帧的特征点与全局地图中的对应特征（边缘线、平面片）进行匹配。
            *   再次使用优化算法（如 LM），最小化当前帧特征点到地图中对应特征的距离，进一步优化当前帧的位姿。
            *   这个过程频率较低（相对于 Odometry），但可以有效修正 Odometry 累积的漂移。
        *   **(可选) 回环检测 (Loop Closure):** 一些 LeGO-LOAM 的变种或后续工作会加入回环检测模块，通过识别之前访问过的场景来进一步消除累积误差，但这在原始 LeGO-LOAM 中可能不是核心部分。
    *   **输出:**
        *   经过地图优化后的、更精确的位姿 (`/aft_mapped_to_init`)。
        *   构建的全局点云地图 (`/map`) 或特征地图。

4.  **`transformFusion` 节点 (位姿融合与发布):**
    *   **输入:** `featureAssociation` 输出的高频里程计位姿，`mapOptmization` 输出的低频优化位姿。
    *   **核心功能:** 融合这两个位姿信息。通常是将低频的精确位姿作为基准，然后用高频的里程计增量来插值，得到一个平滑、连续、高频发布的最终位姿估计。
    *   **输出:** 最终的车辆位姿 (`/integrated_to_init` 或 TF 变换)。

**核心算法与技术总结:**

1.  **范围图像 (Range Image):** 高效处理结构化激光雷达数据（如 Velodyne）的基础。
2.  **地面分割:** 关键优化，显著减少计算量，移除不稳定特征。
3.  **点云聚类:** 过滤噪声，聚焦于有意义的物体。
4.  **曲率计算:** 用于区分边缘点和平面点。
5.  **特征关联 (点到线/点到面):** LOAM 系列算法的核心匹配方式。
6.  **两步位姿优化:** LeGO-LOAM 的特色，分离优化不同自由度，提高效率和精度。
7.  **迭代优化算法 (Levenberg-Marquardt):** 用于求解非线性最小二乘问题（位姿估计）。
8.  **Scan-to-Scan & Scan-to-Map 框架:** LOAM 的经典框架，高频里程计+低频地图优化。

**代码阅读建议:**

1.  **从 `imageProjection` 开始:** 理解预处理流程，特别是地面分割和聚类是如何实现的。
2.  **接着看 `featureAssociation`:** 关注特征提取（曲率计算、点选择）和两步优化（如何分别使用平面/地面点和边缘点进行匹配和优化）。
3.  **然后是 `mapOptmization`:** 理解地图如何构建和维护，以及 Scan-to-Map 匹配如何修正位姿。
4.  **最后看 `transformFusion`:** 理解如何融合高频和低频位姿。
5.  **关注 ROS 接口:** 理解各个节点之间的输入输出（订阅和发布的 Topic），以及它们传递的数据结构 (`sensor_msgs/PointCloud2`, 自定义消息如 `cloud_info`, `geometry_msgs/PoseStamped`, `nav_msgs/Odometry`, TF)。
6.  **查看 `launch` 文件和 `config` 文件:** 这些文件定义了节点如何启动以及各种参数（如特征提取的阈值、优化迭代次数等），理解参数对算法性能的影响很重要。
7.  **核心库依赖:** LeGO-LOAM 大量使用了 PCL (Point Cloud Library) 进行点云处理，以及 Eigen 进行线性代数运算。熟悉这两个库会对理解代码很有帮助。

总的来说，LeGO-LOAM 的源代码体现了在经典 LOAM 基础上，针对特定应用场景（地面车辆）进行巧妙优化的思想，通过地面分割和两步优化等手段，在保证精度的同时显著降低了计算复杂度。
</details>


# VINS-Mono视觉slam实现
数据集[The VIODE dataset](https://github.com/kminoda/VIODE/tree/master)

视觉slam算法[VINS-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono)



# 使用wsl环境开发深度相机（已弃用）
{% note danger modern %}
此部分内容已经弃用，不打算使用深度相机了，但该过程未来仍有可能有参考价值，故保留。以及在wsl环境下即使挂载了usb设备，intel驱动程序也会报错检测不到深度相机，这一问题还没有解决。
{% endnote %}

目前还在寒假期间没有实物制作的条件，打算先跑跑仿真，准备好技术路线。目前打算在windows系统中使用wsl2的ubuntu20进行开发

<details>
<summary>wsl内安装ros后无法启动roscore的问题</summary>
```
triority@Triority-Desktop:~$ roscore
... logging to /home/triority/.ros/log/499be312-e486-11ef-add2-75fb32d30971/roslaunch-Triority-Desktop-36755.log
Checking log directory for disk usage. This may take a while.
Press Ctrl-C to interrupt
Done checking log file disk usage. Usage is 1GB.
```
启动之后卡死在这里。主要原因是虚拟机内网络问题。$ROS_MASTER_URI通常是http://localhost:11311，$ROS_IP通常是WSL的IP地址，这里没有显示，指定为本机IP后问题解决：
```
triority@Triority-Desktop:~$ echo $ROS_MASTER_URI
http://localhost:11311
triority@Triority-Desktop:~$ echo $ROS_IP

triority@Triority-Desktop:~$ export ROS_IP=192.168.0.100
triority@Triority-Desktop:~$ echo $ROS_IP
192.168.0.100

triority@Triority-Desktop:~$ roscore
... logging to /home/triority/.ros/log/f7cd5362-e486-11ef-add2-75fb32d30971/roslaunch-Triority-Desktop-36776.log
Checking log directory for disk usage. This may take a while.
Press Ctrl-C to interrupt
Done checking log file disk usage. Usage is <1GB.

started roslaunch server http://192.168.0.100:42277/
ros_comm version 1.17.0


SUMMARY
========

PARAMETERS
 * /rosdistro: noetic
 * /rosversion: 1.17.0

NODES

auto-starting new master
process[master]: started with pid [36784]
ROS_MASTER_URI=http://192.168.0.100:11311/

setting /run_id to f7cd5362-e486-11ef-add2-75fb32d30971
process[rosout-1]: started with pid [36794]
started core service [/rosout]
^C[rosout-1] killing on exit
[master] killing on exit
shutting down processing monitor...
... shutting down processing monitor complete
done
```

</details>

创建ros的工作空间之后在src文件夹新建一个包用于试验代码：
```
catkin_create_pkg lab roscpp rospy std_msgs
```


**wsl挂载usb设备：**

使用usb3.2连接电脑，要在wsl中使用usb设备请参考[微软的文档](https://learn.microsoft.com/zh-cn/windows/wsl/connect-usb)。

首先安装[USBIPD-WIN项目](https://github.com/dorssel/usbipd-win/releases)

在windows powershell中使用`usbipd list`查看usb设备列表，然后根据总线ID共享设备`usbipd bind --busid 4-4`
```
PS C:\Users\Triority> usbipd list
Connected:
BUSID  VID:PID    DEVICE                                                        STATE
1-2    1462:7d42  USB 输入设备                                                  Not shared
1-5    1ea7:0064  USB 输入设备                                                  Not shared
1-6    1a2c:9ef4  USB 输入设备                                                  Not shared
1-14   8087:0026  英特尔(R) 无线 Bluetooth(R)                                   Not shared
2-1    8086:0b3a  Intel(R) RealSense(TM) Depth Camera 435i Depth, Intel(R) ...  Not shared

Persisted:
GUID                                  DEVICE

```
接下来应该附加USB设备`usbipd attach --wsl --busid 2-1`，但是报错：
```
PS C:\Users\Triority> usbipd attach --wsl --busid 2-1
usbipd: info: Using WSL distribution 'Ubuntu-20.04' to attach; the device will be available in all WSL 2 distributions.
usbipd: info: Detected networking mode 'virtioproxy'.
usbipd: error: Networking mode 'virtioproxy' is not supported.
```
显然需要修改网络模式，这里改成`mirrored`，这个模式的作用在[微软的文档有介绍](https://learn.microsoft.com/en-us/windows/wsl/networking#mirrored-mode-networking)

也就是配置文件`.wslconfig`内容为
```
[wsl2]
networkingMode=mirrored
```
此时将wsl重启`wsl --shutdown`
```
PS C:\Users\Triority> usbipd attach --wsl --busid 2-1
usbipd: info: Using WSL distribution 'Ubuntu-20.04' to attach; the device will be available in all WSL 2 distributions.
usbipd: info: Detected networking mode 'mirrored'.
usbipd: info: Using IP address 127.0.0.1 to reach the host.
WSL usbip: error: Attach Request for 2-1 failed - Device busy (exported)
usbipd: warning: The device appears to be used by Windows; stop the software using the device, or bind the device using the '--force' option.
usbipd: error: Failed to attach device with busid '2-1'.
```
按照要求强制执行`usbipd bind --busid 2-1 --force`并重启电脑

此时重新`usbipd attach --wsl --busid 2-1`并在ubuntu中查看usb设备
```
triority@Triority-Desktop:~$ lsusb
Bus 002 Device 002: ID 8086:0b3a Intel Corp. Intel(R) RealSense(TM) Depth Camera 435i
Bus 002 Device 001: ID 1d6b:0003 Linux Foundation 3.0 root hub
Bus 001 Device 001: ID 1d6b:0002 Linux Foundation 2.0 root hub
```

Intel D435i相关链接：
+ [商品页面](https://www.intelrealsense.com/depth-camera-d435i/)
+ [datasheet](https://www.intelrealsense.com/download/21345/?tmstv=1697035582)
+ [SDK](https://github.com/IntelRealSense/librealsense/releases)
+ [ros1使用](https://github.com/IntelRealSense/realsense-ros/tree/ros1-legacy)

