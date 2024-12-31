/*
MIT License

Copyright (c) 2024 xingxing

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "linalg.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

// 用于浮点数比较的辅助函数
template<typename T>
bool is_close(const T& a, const T& b, T eps = 1e-10) {
    return std::abs(a - b) < eps;
}

// 用于矩阵比较的辅助函数
template<typename T>
bool matrix_is_close(const linalg::Matrix<T>& a, const linalg::Matrix<T>& b, T eps = 1e-10) {
    if (a.get_rows() != b.get_rows() || a.get_cols() != b.get_cols()) {
        return false;
    }
    for (size_t i = 0; i < a.get_rows(); ++i) {
        for (size_t j = 0; j < a.get_cols(); ++j) {
            if (!is_close(a(i, j), b(i, j), eps)) {
                return false;
            }
        }
    }
    return true;
}

void test_vector_operations() {
    std::cout << "测试向量运算..." << std::endl;

    // 构造函数测试
    linalg::Vector<double> v1 = {1.0, 2.0, 3.0};
    linalg::Vector<double> v2 = {4.0, 5.0, 6.0};
    
    // 加法测试
    auto v3 = v1 + v2;
    assert(is_close(v3[0], 5.0));
    assert(is_close(v3[1], 7.0));
    assert(is_close(v3[2], 9.0));
    
    // 减法测试
    auto v4 = v2 - v1;
    assert(is_close(v4[0], 3.0));
    assert(is_close(v4[1], 3.0));
    assert(is_close(v4[2], 3.0));
    
    // 点积测试
    double dot_product = v1.dot(v2);
    assert(is_close(dot_product, 32.0));  // 1*4 + 2*5 + 3*6 = 32
    
    // 标量乘法测试
    auto v5 = v1 * 2.0;
    assert(is_close(v5[0], 2.0));
    assert(is_close(v5[1], 4.0));
    assert(is_close(v5[2], 6.0));

    std::cout << "向量运算测试通过！" << std::endl;
}

void test_matrix_operations() {
    std::cout << "测试矩阵运算..." << std::endl;

    // 构造函数测试
    linalg::Matrix<double> m1 = {
        {1.0, 2.0},
        {3.0, 4.0}
    };
    
    linalg::Matrix<double> m2 = {
        {5.0, 6.0},
        {7.0, 8.0}
    };
    
    // 加法测试
    auto m3 = m1 + m2;
    assert(is_close(m3(0, 0), 6.0));
    assert(is_close(m3(0, 1), 8.0));
    assert(is_close(m3(1, 0), 10.0));
    assert(is_close(m3(1, 1), 12.0));
    
    // 减法测试
    auto m4 = m2 - m1;
    assert(is_close(m4(0, 0), 4.0));
    assert(is_close(m4(0, 1), 4.0));
    assert(is_close(m4(1, 0), 4.0));
    assert(is_close(m4(1, 1), 4.0));
    
    // 矩阵乘法测试
    auto m5 = m1 * m2;
    assert(is_close(m5(0, 0), 19.0));  // 1*5 + 2*7 = 19
    assert(is_close(m5(0, 1), 22.0));  // 1*6 + 2*8 = 22
    assert(is_close(m5(1, 0), 43.0));  // 3*5 + 4*7 = 43
    assert(is_close(m5(1, 1), 50.0));  // 3*6 + 4*8 = 50
    
    // 转置测试
    auto m6 = m1.transpose();
    assert(is_close(m6(0, 0), 1.0));
    assert(is_close(m6(0, 1), 3.0));
    assert(is_close(m6(1, 0), 2.0));
    assert(is_close(m6(1, 1), 4.0));

    std::cout << "矩阵运算测试通过！" << std::endl;
}

void test_matrix_advanced() {
    std::cout << "测试高级矩阵运算..." << std::endl;

    // 测试矩阵求逆
    linalg::Matrix<double> m = {
        {4.0, 7.0},
        {2.0, 6.0}
    };
    
    auto inv = m.inverse();
    
    // 验证求逆结果：m * m^(-1) 应该等于单位矩阵
    auto identity = m * inv;
    assert(is_close(identity(0, 0), 1.0));
    assert(is_close(identity(0, 1), 0.0));
    assert(is_close(identity(1, 0), 0.0));
    assert(is_close(identity(1, 1), 1.0));
    
    // 测试行列式
    double det = m.determinant();
    assert(is_close(det, 10.0));  // 4*6 - 7*2 = 24 - 14 = 10

    std::cout << "高级矩阵运算测试通过！" << std::endl;
}

void test_error_handling() {
    std::cout << "测试错误处理..." << std::endl;

    try {
        // 测试不匹配维度的向量加法
        linalg::Vector<double> v1 = {1.0, 2.0};
        linalg::Vector<double> v2 = {1.0, 2.0, 3.0};
        auto v3 = v1 + v2;
        assert(false);  // 不应该到达这里
    } catch (const std::runtime_error&) {
        // 预期的异常
    }

    try {
        // 测试非方阵求逆
        linalg::Matrix<double> m(2, 3);
        auto inv = m.inverse();
        assert(false);  // 不应该到达这里
    } catch (const std::runtime_error&) {
        // 预期的异常
    }

    try {
        // 测试不可逆矩阵
        linalg::Matrix<double> m = {
            {1.0, 2.0},
            {2.0, 4.0}
        };
        auto inv = m.inverse();
        assert(false);  // 不应该到达这里
    } catch (const std::runtime_error&) {
        // 预期的异常
    }

    std::cout << "错误处理测试通过！" << std::endl;
}

int main() {
    try {
        test_vector_operations();
        test_matrix_operations();
        test_matrix_advanced();
        test_error_handling();
        
        std::cout << "\n所有测试通过！" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "测试失败：" << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 