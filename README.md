+ # NanoMatrix
+ 
+ ![License](https://img.shields.io/badge/license-MIT-blue.svg)
+ ![C++](https://img.shields.io/badge/C%2B%2B-17-brightgreen.svg)
+ 
+ NanoMatrix 是一个轻量级的 C++ 线性代数库，提供向量和矩阵的基本运算功能。它采用现代 C++ 设计，使用模板实现，支持任意数值类型。
+ 
+ ## 特性
+ 
+ ### 设计特点
+ - 🎯 单头文件实现，易于集成
+ - 🔧 模板实现，支持任意数值类型
+ - 🛡️ 异常安全，运行时错误检查
+ - 🚀 现代 C++ 风格 (C++17)
+ - 📝 完整的单元测试
+ - 📖 Doxygen 文档支持
+ 
+ ### 向量运算
+ - ➕ 向量加减法
+ - 📊 点积运算
+ - ✖️ 标量乘法
+ - 📍 下标访问
+ 
+ ### 矩阵运算
+ - ➕ 矩阵加减法
+ - ✖️ 矩阵乘法
+ - 🔄 矩阵转置
+ - ↩️ 矩阵求逆
+ - 📐 行列式计算
+ - 📊 点乘运算（Hadamard积）
+ 
+ ## 安装
+ 
+ 这是一个仅头文件的库，只需要将 `linalg.hpp` 复制到你的项目中即可。
+ 
+ ```bash
+ # 克隆仓库
+ git clone https://github.com/yourusername/NanoMatrix.git
+ 
+ # 复制头文件到你的项目
+ cp NanoMatrix/linalg.hpp /path/to/your/project/
+ ```
+ 
+ ## 使用方法
+ 
+ ### 基本使用
+ 
+ ```cpp
+ #include "linalg.hpp"
+ #include <iostream>
+ 
+ int main() {
+     // 向量运算示例
+     linalg::Vector<double> v1 = {1.0, 2.0, 3.0};
+     linalg::Vector<double> v2 = {4.0, 5.0, 6.0};
+     
+     auto v3 = v1 + v2;              // 向量加法
+     auto dot = v1.dot(v2);          // 点积
+     auto v4 = v1 * 2.0;             // 标量乘法
+ 
+     // 矩阵运算示例
+     linalg::Matrix<double> m1 = {
+         {1.0, 2.0},
+         {3.0, 4.0}
+     };
+     
+     linalg::Matrix<double> m2 = {
+         {5.0, 6.0},
+         {7.0, 8.0}
+     };
+     
+     auto m3 = m1 + m2;              // 矩阵加法
+     auto m4 = m1 * m2;              // 矩阵乘法
+     auto m5 = m1.transpose();       // 矩阵转置
+     auto m6 = m1.inverse();         // 矩阵求逆
+     auto det = m1.determinant();    // 计算行列式
+ }
+ ```
+ 
+ ### 异常处理
+ 
+ ```cpp
+ try {
+     // 尝试对不匹配维度的向量进行运算
+     linalg::Vector<double> v1 = {1.0, 2.0};
+     linalg::Vector<double> v2 = {1.0, 2.0, 3.0};
+     auto v3 = v1 + v2;  // 将抛出异常
+ } catch (const std::runtime_error& e) {
+     std::cerr << "错误：" << e.what() << std::endl;
+ }
+ ```
+ 
+ ## 构建测试
+ 
+ ```bash
+ # 创建构建目录
+ mkdir build && cd build
+ 
+ # 配置项目
+ cmake ..
+ 
+ # 构建
+ cmake --build .
+ 
+ # 运行测试
+ ./test_linalg
+ ```
+ 
+ ## 系统要求
+ 
+ - C++17 或更高版本
+ - CMake 3.10 或更高版本（仅用于构建测试）
+ - 支持的编译器：
+   - GCC 7+
+   - Clang 5+
+   - MSVC 2017+
+ 
+ ## 文档
+ 
+ 完整的 API 文档可以使用 Doxygen 生成：
+ 
+ ```bash
+ doxygen Doxyfile
+ ```
+ 
+ 生成的文档将位于 `docs/html` 目录中。
+ 
+ ## 贡献
+ 
+ 欢迎提交 Issue 和 Pull Request！在提交 PR 之前，请确保：
+ 
+ 1. 代码符合现代 C++ 风格
+ 2. 添加了适当的单元测试
+ 3. 通过所有现有测试
+ 4. 更新了相关文档
+ 
+ ## 许可证
+ 
+ 本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。
+ 
+ ## 作者
+ 
+ xingxing
+ 
+ ## 版本历史
+ 
+ - v1.0.0 (2025-01)
+   - 初始发布
+   - 基本的向量和矩阵运算
+   - 完整的单元测试
