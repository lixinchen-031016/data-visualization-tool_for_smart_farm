# 数据分析可视化工具

这是一个基于 Streamlit 的数据分析和可视化工具，旨在帮助用户快速分析和可视化 CSV、Excel 和 JSON 格式的数据。

## 功能特点

- 支持多种文件格式：CSV、Excel (xlsx/xls)、JSON
- 数据概览：快速查看数据结构和基本统计信息
- 数据清洗：处理缺失值、删除重复行、交互式数据编辑
- 描述性统计：生成数据的统计摘要
- 数据可视化：支持多种图表类型（散点图、线图、柱状图、箱线图、直方图、饼图、热力图）
- 高级分析：相关性分析、数据分组和聚合
- 数据导出：支持导出为 CSV 或 Excel 格式

## 安装说明

1. 克隆此仓库：
   ```bash
   git clone https://github.com/yourusername/data-analysis-tool.git
   ```

2. 进入项目目录：
   ```bash
   cd data-analysis-tool
   ```

3. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

## 使用方法

1. 在项目目录下运行以下命令启动应用：
   ```bash
   streamlit run app.py
   ```

2. 在浏览器中打开显示的 URL（通常是 http://localhost:8501）。

3. 使用侧边栏导航不同的功能页面：
   - 数据概览：上传和预览数据
   - 数据清洗：处理数据质量问题
   - 数据分析：查看描述性统计和相关性分析
   - 可视化：创建各种图表
   - 高级分析：进行更深入的数据探索

## 贡献指南

我们欢迎并感谢任何形式的贡献。如果您想为这个项目做出贡献，请遵循以下步骤：

1. Fork 这个仓库
2. 创建您的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 将您的更改推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启一个 Pull Request

## 许可证

本项目采用 MIT 许可证。详情请见 [LICENSE](LICENSE) 文件。

## 联系方式

如果您有任何问题或建议，请通过以下方式联系我们：

- 项目链接：[https://github.com/yourusername/data-analysis-tool](https://github.com/yourusername/data-analysis-tool)
- 问题反馈：[https://github.com/yourusername/data-analysis-tool/issues](https://github.com/yourusername/data-analysis-tool/issues)

## 致谢

- [Streamlit](https://streamlit.io/) - 用于构建数据应用的开源框架
- [Plotly](https://plotly.com/) - 交互式图表库
- [Pandas](https://pandas.pydata.org/) - 数据处理和分析库

感谢所有为这个项目做出贡献的开发者和用户。
