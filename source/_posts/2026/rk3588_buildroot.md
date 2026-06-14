---
title: 自定义硬件RK3588核心板的buildroot系统与应用交叉编译开发
cover: /img/RUN.png
categories:
  - 文档&笔记
date: 2026-06-09 21:18:59
description: 首先使用正点原子的rk3588核心板+底板进行系统编译，然后裁剪并自制底板并运行buildroot系统
---
# buildroot系统编译
## 环境准备
+ 在vmware中安装ubuntu20.04桌面端系统，作为交叉编译环境。
+ 安装必要依赖
  ```
  sudo apt-get update && sudo apt-get install git ssh make gcc libssl-dev liblz4-tool expect expect-dev g++ patchelf chrpath gawk texinfo chrpath diffstat binfmt-support qemu-user-static live-build bison flex fakeroot cmake gcc-multilib g++-multilib unzip device-tree-compiler ncurses-dev bzip2 expat gpgv2 cpp-aarch64-linux-gnu libgmp-dev libmpc-dev bc python-is-python3 python2 
  ```
+ 将SDK复制到虚拟机内，解压并释放文件
  ```
  mkdir ~/rk3588_linux_sdk 
  tar xvf atk-rk3588_linux_release_v1.2_20250104.tgz -C ~/rk3588_linux_sdk 

  cd ~/rk3588_linux_sdk/ 
  .repo/repo/repo sync -l -j10 
  ```

## SDK 工程目录
每个目录或其子目录会对应一个git工程；因为SDK的代码和相关文档被划分成了若干git 仓库分别进行版本管理。

+ app：存放上层应用app，包括Qt应用程序，以及其它的C/C++应用程序。 
+ buildroot：基于buildroot 开发的根文件系统。 
+ debian：基于Debian开发的根文件系统。 
+ device/rockchip：存放各芯片板级配置文件和Parameter分区表文件，以及一些编译与打包固件的脚本和预备文件。 
+ docs：存放芯片模块开发指导文档、平台支持列表、芯片平台相关文档、Linux开发指南等。 
+ external：存放所需的第三方库，包括音频、视频、网络、recovery等。 
+ kernel：Linux 5.10 版本内核源码。 
+ prebuilts：存放交叉编译工具链。 
+ rkbin：存放Rockchip相关的Binary和工具。 
+  rockdev：存放编译输出固件，编译SDK后才会生成该文件夹。 
+ tools：存放Linux和Windows操作系统环境下常用的工具，包括镜像烧录工具、SD卡升级启动制作工具、批量烧录工具等，譬如前面给大家介绍的 RKDevTool 工具以及Linux_Upgrade_Tool 工具都存放在该目录。 
+ u-boot：基于v2017.09版本进行开发的uboot源码。 
+ yocto：基于Yocto开发的根文件系统

## 编译
```shell
triority@ubuntu:~/rk3588_linux_sdk$ ./build.sh lunch 

############### Rockchip Linux SDK ###############

Manifest: atk-rk3588_linux5.10_release_v1.2_20250104.xml

Log saved at /home/triority/rk3588_linux_sdk/output/sessions/2026-06-09_07-27-39

Pick a defconfig:

1. rockchip_defconfig
2. alientek_rk3588_defconfig
3. rockchip_rk3588_evb1_lp4_v10_defconfig
4. rockchip_rk3588_evb7_v11_defconfig
5. rockchip_rk3588s_evb1_lp4x_v10_defconfig
Which would you like? [1]: 2
Switching to defconfig: /home/triority/rk3588_linux_sdk/device/rockchip/.chip/alientek_rk3588_defconfig
make: Entering directory '/home/triority/rk3588_linux_sdk/device/rockchip/common'
mkdir -p /home/triority/rk3588_linux_sdk/output/kconf/lxdialog
make CC="gcc" HOSTCC="gcc" \
    obj=/home/triority/rk3588_linux_sdk/output/kconf -C /home/triority/rk3588_linux_sdk/device/rockchip/common/kconfig -f Makefile.br conf
make[1]: Entering directory '/home/triority/rk3588_linux_sdk/device/rockchip/common/kconfig'
gcc -D_DEFAULT_SOURCE -D_XOPEN_SOURCE=600 -DCURSES_LOC="<ncurses.h>" -DNCURSES_WIDECHAR=1 -DLOCALE  -I/home/triority/rk3588_linux_sdk/output/kconf -DCONFIG_=\"\"  -MM *.c > /home/triority/rk3588_linux_sdk/output/kconf/.depend 2>/dev/null || :
gcc -D_DEFAULT_SOURCE -D_XOPEN_SOURCE=600 -DCURSES_LOC="<ncurses.h>" -DNCURSES_WIDECHAR=1 -DLOCALE  -I/home/triority/rk3588_linux_sdk/output/kconf -DCONFIG_=\"\"   -c conf.c -o /home/triority/rk3588_linux_sdk/output/kconf/conf.o
gcc -D_DEFAULT_SOURCE -D_XOPEN_SOURCE=600 -DCURSES_LOC="<ncurses.h>" -DNCURSES_WIDECHAR=1 -DLOCALE  -I/home/triority/rk3588_linux_sdk/output/kconf -DCONFIG_=\"\"  -I. -c /home/triority/rk3588_linux_sdk/output/kconf/zconf.tab.c -o /home/triority/rk3588_linux_sdk/output/kconf/zconf.tab.o
gcc -D_DEFAULT_SOURCE -D_XOPEN_SOURCE=600 -DCURSES_LOC="<ncurses.h>" -DNCURSES_WIDECHAR=1 -DLOCALE  -I/home/triority/rk3588_linux_sdk/output/kconf -DCONFIG_=\"\"   /home/triority/rk3588_linux_sdk/output/kconf/conf.o /home/triority/rk3588_linux_sdk/output/kconf/zconf.tab.o  -o /home/triority/rk3588_linux_sdk/output/kconf/conf
rm /home/triority/rk3588_linux_sdk/output/kconf/zconf.tab.c
make[1]: Leaving directory '/home/triority/rk3588_linux_sdk/device/rockchip/common/kconfig'
#
# configuration written to /home/triority/rk3588_linux_sdk/output/.config
#
make: Leaving directory '/home/triority/rk3588_linux_sdk/device/rockchip/common'
```

build脚本的全部功能：
```shell
triority@ubuntu:~/rk3588_linux_sdk$ ./build.sh -h 

############### Rockchip Linux SDK ###############

Manifest: atk-rk3588_linux5.10_release_v1.2_20250104.xml

Usage: build.sh [OPTIONS]
Available options:
chip[:<chip>[:<config>]]          	choose chip
defconfig[:<config>]              	choose defconfig
 *_defconfig                      	switch to specified defconfig
    available defconfigs:
	alientek_rk3588_defconfig
	rockchip_defconfig
	rockchip_rk3588_evb1_lp4_v10_defconfig
	rockchip_rk3588_evb7_v11_defconfig
	rockchip_rk3588s_evb1_lp4x_v10_defconfig
 olddefconfig                     	resolve any unresolved symbols in .config
 savedefconfig                    	save current config to defconfig
 menuconfig                       	interactive curses-based configurator
config                            	modify SDK defconfig
shell                             	setup a shell for developing
print-parts                        	print partitions
mod-parts                          	interactive partition table modify
edit-parts                         	edit raw partitions
new-parts:<offset>:<name>:<size>...	re-create partitions
insert-part:<idx>:<name>[:<size>]  	insert partition
del-part:(<idx>|<name>)            	delete partition
move-part:(<idx>|<name>):<idx>     	move partition
rename-part:(<idx>|<name>):<name>  	rename partition
resize-part:(<idx>|<name>):<size>  	resize partition
kernel[:cmds]                    	build kernel
modules[:cmds]                   	build kernel modules
linux-headers[:cmds]             	build linux-headers
kernel-config[:cmds]             	modify kernel defconfig
kernel-make[:<arg1>:<arg2>]      	run kernel make (alias kmake)
wifibt[:<dst dir>[:<chip>]]       	build Wifi/BT
rtos                             	build and pack RTOS
buildroot-config[:<config>]       	modify buildroot defconfig
buildroot-make[:<arg1>:<arg2>]    	run buildroot make (alias bmake)
rootfs[:<rootfs type>]            	build default rootfs
buildroot                         	build buildroot rootfs
yocto                             	build yocto rootfs
debian                            	build debian rootfs
recovery                          	build recovery
pcba                              	build PCBA
security_check                    	check contidions for security boot
createkeys                        	build security boot keys
security_ramboot                  	build security ramboot
security_uboot                    	build uboot with security
security_boot                     	build boot with security
security_recovery                 	build recovery with security
security_rootfs                   	build rootfs with security
loader[:cmds]                    	build loader (uboot)
uboot[:cmds]                     	build u-boot
uefi[:cmds]                      	build uefi
firmware                          	pack and check firmwares
edit-package-file                 	edit package-file
edit-ota-package-file             	edit A/B OTA package-file
updateimg                         	build update image
otapackage                        	build A/B OTA update image
all                               	build all images
save                              	save images and build info
allsave                           	build all images and save them
cleanall                          	cleanup
clean[:module[:module]]...        	cleanup modules
    available modules:
	all
	config
	firmware
	kernel
	loader
	pcba
	recovery
	rootfs
	updateimg
post-rootfs <rootfs dir>          	trigger post-rootfs hook scripts
help                              	usage

Default option is 'allsave'.
```

其中常用的包括：

lunch 选择板级配置文件： `./build.sh lunch `
uboot 编译u-boot： `./build.sh uboot `
kernel 编译kernel： `./build.sh kernel `
modules 编译内核模块： `./build.sh modules `
rootfs 编译根文件系统： `./build.sh rootfs `
buildroot 编译buildroot根文件系统： `./build.sh buildroot `
debian 编译Debian根文件系统： `./build.sh debian `
recovery 编译recovery： `./build.sh recovery `
all 编译整个SDK，包括uboot、kernel、rootfs、recovery： `./build.sh all `
cleanall 清理整个SDK： `./build.sh cleanall `
firmware 将镜像打包到rockdev目录： `./build.sh firmware `
updateimg 将所有镜像打包成一个update.img固件：`./build.sh updateimg `

执行“./build.sh all”编译完成后，会自动将所有分立镜像打包成update.img

## SDK开发
### SDK板级配置
SDK板级配置文件存放在`<SDK_PATH>/device/rockchip/rk3588/`目录。该目录下存在多个`*_defconfig`配置文件，对应`./build.sh lunch`命令打印的配置列表。正点原子ATK-DLRK3588开发板对应的配置文件为`alientek_rk3588_defconfig`。该文件中定义了ATK-DLRK3588平台所使用的内核设备树文件以及其它配置信息；该文件中只是定义了一小部分配置信息，绝大部分配置信息保持默认。 我们也可在SDK根目录下执行“make menuconfig”命令打开图形化配置界面、对配置进行更改。

`alientek_rk3588_defconfig`文件内容为：
```conf
RK_YOCTO_CFG="rockchip-rk3588-evb"
RK_KERNEL_DTS_NAME="rk3588-atk-devkit"
RK_USE_FIT_IMG=y
RK_TARGET_BOARD="ATK_DLRK3588"
RK_BUILD_JOBS="24"
RK_BUILDROOT_BASE_CFG="atk_dlrk3588"
RK_ROOTFS_HOSTNAME_CUSTOM=y
RK_ROOTFS_HOSTNAME="ATK-DLRK3588"
```
最关键的一项`RK_KERNEL_DTS_NAME="rk3588-atk-devkit"`，它决定内核使用哪份设备树，描述了板子的硬件连接。

### U-boot
对于U-Boot，RK3588平台所使用的设备树文件为`arch/arm/dts/rk3588-evb.dts`。U-Boot设备树负责初始化存储、调试串口等基础外设；而kernel设备树初始化存储、调试串口之外的外设，譬如LCD显示、千兆网等。执行U-Boot代码时先用U-Boot的设备树完成存储、调试串口等基础外设的初始化操作，然后从存储上加载kernel的设备树并转而使用这份设备树继续初始化其余外设。 

### kernel
Linux 内核源码存放在`<SDK_PATH>/kernel/`目录

对于Linux内核，正点原子ATK-DLRK3588平台所使用的defconfig配置文件为`arch/arm64/configs/rockchip_linux_defconfig`。Rockchip平台使用的设备树文件都存放在`arch/arm64/boot/dts/rockchip/`目录下。正点原子ATK-DLRK3588平台对应的设备树文件为`arch/arm64/boot/dts/rockchip/rk3588-atk-devkit.dts`。`rk3588-atk-devkit.dts`作为顶层设备树文件、其包含有多个`.dtsi`设备树，其结构如下：

+ rk3588-atk-devkit.dts 
  + rk3588-atk-devkit.dtsi 
    + rk3588.dtsi 
    + rk3588-evb.dtsi 
    + rk3588-rk806-single.dtsi 
  + rk3588-atk-cameras.dtsi 
  + rk3588-atk-screen_choose.dtsi 
  + rk3588-atk-lcds.dtsi 
  + rk3588-linux.dtsi 

+ rk3588.dtsi：该设备树文件是RK3588平台级设备树文件，与具体板级硬件无关，纯SoC级别的设备树文件，由RK提供，开发者无需改动该文件！ 
+ rk3588-evb.dtsi：RK3588平台板级通用设备树文件，通常被板级设备树文件所包含； 
+ rk3588-rk806-single.dtsi：rk806（pmic）相关配置信息； 
+ rk3588-linux.dtsi：该设备树包含Linux部分特有配置信息（与Android区分）； 
+ rk3588-atk-devkit.dtsi：板级设备树文件，该设备树文件包含板级通用设备树文件rk3588-evb.dtsi以及RK3588平台级设备树文件rk3588.dtsi；
+ rk3588-atk-screen_choose.dtsi：该设备树用于选择需要使能的LCD屏，譬如HDMI、MIPI、DP等； 
+ rk3588-atk-lcds.dtsi：该设备树包含LCD相关的配置信息，譬如触摸屏、背光、LCD时序参数、显示接口等配置信息； 
+ rk3588-atk-cameras.dtsi：该设备树包含开发板的摄像头相关配置信息。


### buildroot
Buildroot 源码存放在`<SDK_PATH>/buildroot` 目录下

编译RK3588 Linux SDK过程中，需要编译两次buildroot：
+ 编译buildroot得到rootfs.img。rootfs.img 会烧录到开发板 rootfs分区，作为正常启动模式下挂载的根文件系统；
+ 编译buildroot得到ramdisk（这也是根文件系统，进入recovery模式时挂载的小体积根文件系统，而 rootfs.img则是正常启动模式下挂载的根文件系统）。ramdisk 最终会打包进recovery.img 镜像中，recovery.img 会烧录到开发板recovery分区。

buildroot根目录下的文件夹：
+ arch	存放 Buildroot 支持的所有 CPU 架构相关的配置文件及构建脚本。
+ board	存放特定目标平台相关的文件，例如内核配置、补丁文件、rootfs 覆盖文件等。
+ boot	存放 Buildroot 支持的 BootLoader 相关补丁、校验文件、构建脚本和配置选项等。
+ build	Buildroot 编译系统相关组件。
+ configs	存放所有目标平台的 defconfig 配置文件。
+ dl	download 的缩写，用于存放下载的各种第三方开源软件包，例如 alsa-lib、bluez、bzip2、curl 等。在编译过程中，Buildroot 会从网络下载所需软件包，并放到 dl/ 目录。如果下载失败，也可以手动下载对应软件包并复制到该目录，后续编译时就不会重复下载。
+ docs	存放相关参考文档。
+ fs	存放各种文件系统相关的源码。
+ linux	存放 Linux 内核的构建脚本和配置选项。
+ output	编译 Buildroot 后生成的输出目录，用于存放编译过程中的各种文件，包括中间目标文件、可执行文件、库文件以及最终烧录到开发板的 rootfs 镜像等。
+ package	存放所有软件包的构建脚本和配置选项。每个软件包通常在 `package/<package_name>/` 目录下包含一个 Config.in 文件和一个 `<package_name>.mk` 文件，其中 `.mk` 文件本质上是 Makefile 文件。如果需要新增软件包，通常需要在 package/ 目录下添加相应内容。
+ support	存放 Buildroot 提供功能支持所需的脚本、配置文件等。
+ toolchain	存放制作交叉编译工具链的构建脚本和相关文件，例如 binutils、gcc、gdb、kernel-header 和 uClibc 等。
+ utils	存放 Buildroot 的实用脚本和工具。

#### 编译命令
+ 编译根文件系统
    命令中最后一个参数（rockchip_atk_dlrk3588）用于指定编译目标平台，所有平台的defconfig配置文件都存放在configs目录下，在该目录下可以找到正点原子ATK-DLRK3588平台对应的配置文件rockchip_atk_dlrk3588_defconfig
    ```
    source build/envsetup.sh rockchip_atk_dlrk3588 
    make -j16
    ```
+ 编译package（编译特定软件包）
    设置环境变量
    ```
    source build/envsetup.sh rockchip_atk_dlrk3588 
    ```
    编译指定的package
    ```
    make <package_name>
    ```
#### rootfs配置
在buildroot源码目录下，执行如下命令打开menuconfig图形化配置界面
```shell
source build/envsetup.sh rockchip_atk_dlrk3588  //设置环境变量 
make menuconfig        //打开menuconfig图形化配置界面
```
修改完成后保存配置：
```shell
make savedefconfig
```
接下来回到SDK根目录下运行build.sh脚本编译buildroot：
```shell
cd ../     //回到SDK根目录下（当前处于buildroot源码目录下） 
./build.sh buildroot  //执行build.sh脚本编译buildroot 
```

### 烧录
这里我在rootfs加入了一些新的包，包括python-flask和v4l2等。成功编译镜像之后准备下载

rootfs镜像文件保存于：
```
buildroot/output/rockchip_atk_dlrk3588/images/rootfs.ext2
```
烧录需要和parameter.txt一起重新烧录：
```
rk3588_linux_sdk/device/rockchip/rk3588/parameter.txt
```
注意不要使用软连接地址，必须使用上面的实际文件地址。

# buildroot内应用开发
## 一些依赖问题
v4l2-ctl和opencv等库启动报错缺少库文件`libgcc_s.so.1`，这里直接通过overlay的方式编译进系统里

在 Ubuntu 上执行寻找库文件，注意检查cpu架构是否是`ARM aarch64`
```
triority@ubuntu:~/rk3588_linux_sdk$ find buildroot/output/rockchip_atk_dlrk3588 -name "libgcc_s.so.1" -print
buildroot/output/rockchip_atk_dlrk3588/build/gcc-target-10.4.0/host-aarch64-buildroot-linux-gnu/gcc/libgcc_s.so.1
buildroot/output/rockchip_atk_dlrk3588/build/gcc-target-10.4.0/aarch64-buildroot-linux-gnu/libgcc/libgcc_s.so.1
buildroot/output/rockchip_atk_dlrk3588/build/camera-engine-rkaiq-1.0/rkisp_demo/demo/iio/lib/libgcc_s.so.1
buildroot/output/rockchip_atk_dlrk3588/build/host-gcc-final-10.4.0/build/gcc/libgcc_s.so.1
buildroot/output/rockchip_atk_dlrk3588/build/host-gcc-final-10.4.0/build/aarch64-buildroot-linux-gnu/libgcc/libgcc_s.so.1
buildroot/output/rockchip_atk_dlrk3588/host/aarch64-buildroot-linux-gnu/lib64/libgcc_s.so.1
buildroot/output/rockchip_atk_dlrk3588/host/aarch64-buildroot-linux-gnu/sysroot/usr/lib/libgcc_s.so.1
triority@ubuntu:~/rk3588_linux_sdk$ file buildroot/output/rockchip_atk_dlrk3588/host/aarch64-buildroot-linux-gnu/sysroot/usr/lib/libgcc_s.so.1
buildroot/output/rockchip_atk_dlrk3588/host/aarch64-buildroot-linux-gnu/sysroot/usr/lib/libgcc_s.so.1: ELF 64-bit LSB shared object, ARM aarch64, version 1 (SYSV), dynamically linked, with debug_info, not stripped

```
然后放到 `/usr/lib`内：
```
triority@ubuntu:~/rk3588_linux_sdk$ mkdir -p rootfs-overlay/usr/lib
triority@ubuntu:~/rk3588_linux_sdk$ 
triority@ubuntu:~/rk3588_linux_sdk$ cp -a buildroot/output/rockchip_atk_dlrk3588/host/aarch64-buildroot-linux-gnu/sysroot/usr/lib/libgcc_s.so.1 \
> rootfs-overlay/usr/lib/
triority@ubuntu:~/rk3588_linux_sdk$ ls -lh rootfs-overlay/usr/lib/libgcc_s.so.1
-rw-r--r-- 1 triority triority 562K Jun  9 08:12 rootfs-overlay/usr/lib/libgcc_s.so.1

```
在 Buildroot 里查看 overlay 路径，在`make menuconfig`中：
```
System configuration
  Root filesystem overlay directories
```
如果已经有配置就把需要添加的文件复制进去，如果没有有设置为自己的路径。

我这里已经有了，因此把文件复制进去
```
mkdir -p board/rockchip/rk3588/fs-overlay/usr/lib

cp -a output/rockchip_atk_dlrk3588/host/aarch64-buildroot-linux-gnu/sysroot/usr/lib/libgcc_s.so.1 board/rockchip/rk3588/fs-overlay/usr/lib/
```
然后重新编译rootfs并下载，现在不会再报这个错误了

## 自定义应用开发
### 交叉编译工具
项目中需要实现的任务是使用npu运行ai模型对图像进行实时增强+web实时预览回放和参数控制。

为了极限性能项目计划改用使用C/C++开发

首先要找到交叉编译器的位置，然后可以作为环境变量便于在SDK外部的项目代码调用
```shell
triority@ubuntu:~/rk3588_linux_sdk$ ls buildroot/output/rockchip_atk_dlrk3588/host/bin/*gcc
buildroot/output/rockchip_atk_dlrk3588/host/bin/aarch64-buildroot-linux-gnu-gcc  buildroot/output/rockchip_atk_dlrk3588/host/bin/aarch64-linux-gcc
triority@ubuntu:~/rk3588_linux_sdk$ ls buildroot/output/rockchip_atk_dlrk3588/host/bin/*g++
buildroot/output/rockchip_atk_dlrk3588/host/bin/aarch64-buildroot-linux-gnu-g++  buildroot/output/rockchip_atk_dlrk3588/host/bin/aarch64-linux-g++
```
设置交叉编译环境:
```shell
export SDK=~/rk3588_linux_sdk
export BR_OUT=$SDK/buildroot/output/rockchip_atk_dlrk3588
export PATH=$BR_OUT/host/bin:$PATH
export CROSS_COMPILE=aarch64-buildroot-linux-gnu-

which aarch64-buildroot-linux-gnu-gcc
which aarch64-buildroot-linux-gnu-g++
```
可以看到目前交叉编译器的路径：
```
~/rk3588_linux_sdk/buildroot/output/rockchip_atk_dlrk3588/host/bin/aarch64-buildroot-linux-gnu-gcc
~/rk3588_linux_sdk/buildroot/output/rockchip_atk_dlrk3588/host/bin/aarch64-buildroot-linux-gnu-g++
```
然后就可以做一次交叉编译然后上传开发板运行的测试了:
```shell
cd ~/rk3588-edge-vision-enhancer/rk_vision
mkdir -p src
```
```shell
cat > src/main.cpp <<'EOF'
#include <iostream>

int main() {
    std::cout << "Hello World!" << std::endl;
    return 0;
}
EOF
```
编译并上传：
```shell
aarch64-buildroot-linux-gnu-g++ src/main.cpp -o rkvision

scp rkvision root@192.168.144.103:/root/
```
然后ssh连接运行：
```shell
ssh root@192.168.144.103
chmod +x /root/rkvision
/root/rkvision
```
可以看到成功输出`Hello World!`

### Cmake和自动化编译上传测试
使用cmake编译：
`toolchain-rk3588.cmake`是交叉编译工具链配置文件，
```cmake
# 目标系统linux
set(CMAKE_SYSTEM_NAME Linux)
# 目标处理器架构是 ARM64
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# 定义SDK路径和输出路径
set(SDK_ROOT "$ENV{HOME}/rk3588_linux_sdk")
set(BR_OUT "${SDK_ROOT}/buildroot/output/rockchip_atk_dlrk3588")
# 定义交叉编译器所在目录
set(TOOLCHAIN_BIN "${BR_OUT}/host/bin")
# 定义目标系统的 sysroot，即开发板 rootfs 的编译参考环境，编译器将从这里找 ARM64 版本的头文件和库
set(SYSROOT "${BR_OUT}/host/aarch64-buildroot-linux-gnu/sysroot")

# 指定 C/C++ 编译器
set(CMAKE_C_COMPILER "${TOOLCHAIN_BIN}/aarch64-buildroot-linux-gnu-gcc")
set(CMAKE_CXX_COMPILER "${TOOLCHAIN_BIN}/aarch64-buildroot-linux-gnu-g++")
# 指定 sysroot，查找库、头文件、包配置时以这个 sysroot 作为根目录
set(CMAKE_SYSROOT "${SYSROOT}")
set(CMAKE_FIND_ROOT_PATH "${SYSROOT}")

# 查找程序的规则，NEVER 程序工具从主机环境找
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# 查找库、头文件、包的规则，ONLY 只在 sysroot 里找，防止链接到 Ubuntu 主机的 x86_64 库
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# 设置 pkg-config 的 sysroot，让 pkg-config 输出路径时自动加上 sysroot 前缀
set(ENV{PKG_CONFIG_SYSROOT_DIR} "${SYSROOT}")
# 设置 pkg-config 搜索路径，只去 RK3588 sysroot 里找 .pc 文件
set(ENV{PKG_CONFIG_LIBDIR} "${SYSROOT}/usr/lib/pkgconfig:${SYSROOT}/usr/share/pkgconfig")
# 清空 PKG_CONFIG_PATH，避免 pkg-config 找到 Ubuntu 主机上的 .pc 文件
set(ENV{PKG_CONFIG_PATH} "")
```
`CMakeLists.txt`项目编译规则文件：
```cmake
# CMake 最低版本
cmake_minimum_required(VERSION 3.10)

# 项目名称
project(rkvision)

# 设置 C++ 标准，强制要求
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 PkgConfig，REQUIRED 表示必须找到，否则停止配置
find_package(PkgConfig REQUIRED)

# 调用 pkg-config 查找名为 opencv4 的模块
pkg_check_modules(OPENCV REQUIRED opencv4)

# 添加 OpenCV 头文件路径
include_directories(
    ${OPENCV_INCLUDE_DIRS}
)
# 添加 OpenCV 库目录
link_directories(
    ${OPENCV_LIBRARY_DIRS}
)
# 生成可执行程序，表示把src/main.cpp编译成一个可执行程序rkvision
add_executable(rkvision
    src/main.cpp
)

# 链接 OpenCV 库，告诉链接器 rkvision 这个程序需要链接 OpenCV 的库
target_link_libraries(rkvision
    ${OPENCV_LIBRARIES}
)
# 添加额外编译参数，把 OpenCV 额外要求的编译参数加进去，PRIVATE 表示这些选项只作用于 rkvision 这个目标
target_compile_options(rkvision PRIVATE
    ${OPENCV_CFLAGS_OTHER}
)
```

`main.cpp`加入opencv图像读取并保存：
```cpp
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

//主函数入口：argc命令行参数个数；argv命令行参数内容
//例如运行./rkvision /dev/video22则argc = 2，argv[0] = "./rkvision"，argv[1] = "/dev/video22"
int main(int argc, char** argv) {
    std::string device = "/dev/video22";

    if (argc >= 2) {
        device = argv[1];
    }

    std::cout << "Opening camera through GStreamer: " << device << std::endl;

    //构造 GStreamer 管线
    std::string pipeline =
        //使用 GStreamer 的 V4L2 插件从摄像头采集数据
        "v4l2src device=" + device + " ! "
        //摄像头输出原始视频NV12 格式1920x1080 分辨率30 FPS
        "video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1 ! "
        //GStreamer 的格式转换插件把NV12转换成BGR
        "videoconvert ! "
        "video/x-raw,format=BGR ! "
        //appsink 是 GStreamer 管线的出口，把图像交给 C++ 程序。drop=true如果程序处理不过来允许丢弃旧帧；sync=false不按时间戳严格同步尽快把帧交给程序
        "appsink drop=true sync=false";

    std::cout << "Pipeline: " << pipeline << std::endl;

    //创建一个 OpenCV 视频采集对象，使用GStreamer 管线
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);

    if (!cap.isOpened()) {
        std::cerr << "Error: failed to open camera with GStreamer pipeline." << std::endl;
        return 1;
    }

    //创建图像对象
    cv::Mat frame;

    std::cout << "Reading one frame..." << std::endl;

    //读取图像并检查
    if (!cap.read(frame)) {
        std::cerr << "Error: failed to read frame." << std::endl;
        return 2;
    }

    if (frame.empty()) {
        std::cerr << "Error: captured frame is empty." << std::endl;
        return 3;
    }

    std::cout << "Captured frame: "
              << frame.cols << "x" << frame.rows
              << ", channels=" << frame.channels()
              << std::endl;

    std::string save_path = "/root/app/capture.jpg";

    if (!cv::imwrite(save_path, frame)) {
        std::cerr << "Error: failed to save image to " << save_path << std::endl;
        return 4;
    }

    std::cout << "Saved image to: " << save_path << std::endl;

    return 0;
}
```

可以使用sh脚本自动化编译上传和测试：
`deploy.sh`：
```sh
#!/bin/bash
set -e

BOARD_IP=192.168.144.103
BOARD_USER=root
APP_NAME=rkvision
BUILD_DIR=build-rk3588
REMOTE_DIR=/root/app

SSH_TARGET=${BOARD_USER}@${BOARD_IP}

# SSH 连接复用 socket 文件
CONTROL_PATH="/tmp/ssh_mux_%r@%h:%p"

# 重烧系统开发板 host key 会变化。
# 下面两个选项可以避免每次提示 known_hosts 冲突。
# 仅建议在自己的局域网开发板环境使用。
SSH_OPTS=(
    -o ControlMaster=auto
    -o ControlPath=${CONTROL_PATH}
    -o ControlPersist=10m
    -o StrictHostKeyChecking=no
    -o UserKnownHostsFile=/dev/null
)

echo "========== Build =========="

mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=../toolchain-rk3588.cmake \
  -DCMAKE_BUILD_TYPE=Debug

make -j$(nproc)

cd ..

echo "========== Open SSH master connection =========="

ssh "${SSH_OPTS[@]}" -MNf ${SSH_TARGET}

echo "========== Deploy =========="

ssh "${SSH_OPTS[@]}" ${SSH_TARGET} "mkdir -p ${REMOTE_DIR}"

scp "${SSH_OPTS[@]}" ${BUILD_DIR}/${APP_NAME} ${SSH_TARGET}:${REMOTE_DIR}/

echo "========== Run =========="

ssh "${SSH_OPTS[@]}" ${SSH_TARGET} \
    "chmod +x ${REMOTE_DIR}/${APP_NAME} && cd ${REMOTE_DIR} && ./${APP_NAME}"

echo "========== Close SSH master connection =========="

ssh "${SSH_OPTS[@]}" -O exit ${SSH_TARGET} 2>/dev/null || true
```

