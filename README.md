CoinLocator-HoughCannyDIY/
│
├── src/                        # 源代码目录
│   ├── __init__.py             # 初始化Python包
│   ├── image_processing.py     # 图像预处理模块（读取、转换、滤波）
│   ├── edge_detection.py       # 边缘检测模块（梯度计算、非极大值抑制、双阈值处理）
│   ├── circle_detection.py     # 圆检测模块（霍夫圆变换实现）
│   └── visualization.py        # 结果可视化模块（绘制检测到的圆和圆心）
│
├── data/                       # 数据目录，用于存放测试图像等数据文件
│   ├── test_images/            # 测试图像
│   └── results/                # 输出结果，如标记后的图像
│
├── docs/                       # 文档目录
│   ├── setup.md                # 安装和配置指南
│   ├── usage.md                # 使用说明
│   └── development.md          # 开发者文档，包括设计决策和实现细节
│
├── tests/                      # 测试代码
│   ├── __init__.py             # 初始化Python包
│   └── test_functions.py       # 测试各功能模块
│
├── notebooks/                  # Jupyter笔记本，用于演示和实验
│   └── demo.ipynb              # 演示项目功能的笔记本
│
├── .gitignore                  # Git忽略文件配置
├── LICENSE                     # 项目许可证
└── README.md                   # 项目概述，包括安装、使用、贡献指南等
