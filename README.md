CoinLocator-HoughCannyDIY/
│
├── src/                        # 源代码目录
│   ├── __init__.py             # 初始化Python包
│   ├── edge_detection.py       # 边缘检测模块
│   └── circle_detection.py     # 圆检测模块（霍夫圆变换实现）
│
├── data/                       # 数据目录，用于存放测试图像等数据文件
│   ├── test_images/            # 测试图像
│   └── results/                # 输出结果，如标记后的图像
│
├── docs/                       # 文档目录
│   └── development.md          # 实验报告
│
├── test.ipynb                  # 测试函数
│
└── README.md                   # 项目概述，包括安装、使用、贡献指南等


test.ipynb中第一个格子是调用手动实现的函数，第二个格子是调用cv2的函数（easy version）