# crossroad_detect
复现论文《基于航拍图像的目标检测系统设计与实现》的后半部分，效果良好，但很依赖调参和图像选择
论文经过预处理、二值化、霍夫变换检测出直线，寻找直线交点。对交点聚类实现十字路口的粗定位，再使用创新的离散矩形簇来判别形状。
思路比较新奇，速度尚可。
