/**
 * @file linalg.hpp
 * @brief 轻量级线性代数库
 * @author xingxing
 * @date 2025
 * 
 * 这是一个头文件库，提供基本的向量和矩阵运算功能。
 * 使用模板实现，支持任意数值类型。
 */

#pragma once

#include <vector>
#include <initializer_list>
#include <stdexcept>
#include <iostream>
#include <cmath>

namespace linalg {

/**
 * @brief 向量类
 * @tparam T 数值类型
 * 
 * 提供基本的向量运算功能，包括加减法、点积和标量乘法。
 */
template<typename T>
class Vector {
private:
    std::vector<T> data;  ///< 存储向量数据

public:
    /** @brief 默认构造函数 */
    Vector() = default;
    
    /**
     * @brief 指定大小构造函数
     * @param size 向量维度
     */
    Vector(size_t size) : data(size) {}
    
    /**
     * @brief 初始化列表构造函数
     * @param list 初始化数据
     */
    Vector(std::initializer_list<T> list) : data(list) {}

    /**
     * @brief 访问向量元素
     * @param i 索引
     * @return 对元素的引用
     */
    T& operator[](size_t i) noexcept { return data[i]; }
    const T& operator[](size_t i) const noexcept { return data[i]; }
    
    /**
     * @brief 获取向量维度
     * @return 向量维度
     */
    size_t size() const noexcept { return data.size(); }

    /**
     * @brief 向量加法
     * @param rhs 右操作数
     * @return 和向量
     * @throw std::runtime_error 维度不匹配时抛出
     */
    Vector operator+(const Vector& rhs) const {
        if (size() != rhs.size()) {
            throw std::runtime_error("向量维度不匹配");
        }
        Vector result(size());
        for (size_t i = 0; i < size(); ++i) {
            result[i] = data[i] + rhs[i];
        }
        return result;
    }

    /**
     * @brief 向量减法
     * @param rhs 右操作数
     * @return 差向量
     * @throw std::runtime_error 维度不匹配时抛出
     */
    Vector operator-(const Vector& rhs) const {
        if (size() != rhs.size()) {
            throw std::runtime_error("向量维度不匹配");
        }
        Vector result(size());
        for (size_t i = 0; i < size(); ++i) {
            result[i] = data[i] - rhs[i];
        }
        return result;
    }

    /**
     * @brief 计算向量点积
     * @param rhs 右操作数
     * @return 点积结果
     * @throw std::runtime_error 维度不匹配时抛出
     */
    T dot(const Vector& rhs) const {
        if (size() != rhs.size()) {
            throw std::runtime_error("向量维度不匹配");
        }
        T result = T();
        for (size_t i = 0; i < size(); ++i) {
            result += data[i] * rhs[i];
        }
        return result;
    }

    /**
     * @brief 向量标量乘法
     * @param scalar 标量值
     * @return 乘积向量
     */
    Vector operator*(const T& scalar) const {
        Vector result(size());
        for (size_t i = 0; i < size(); ++i) {
            result[i] = data[i] * scalar;
        }
        return result;
    }
};

/**
 * @brief 矩阵类
 * @tparam T 数值类型
 * 
 * 提供基本的矩阵运算功能，包括加减法、矩阵乘法、转置、求逆等。
 */
template<typename T>
class Matrix {
private:
    std::vector<std::vector<T>> data;  ///< 存储矩阵数据
    size_t rows;  ///< 行数
    size_t cols;  ///< 列数

public:
    /**
     * @brief 构造指定大小的矩阵
     * @param r 行数
     * @param c 列数
     */
    Matrix(size_t r, size_t c) : data(r, std::vector<T>(c)), rows(r), cols(c) {}
    /**
     * @brief 使用初始化列表构造矩阵
     * @param list 二维初始化列表
     * @throw std::runtime_error 当矩阵为空或行长度不一致时抛出
     */
    Matrix(std::initializer_list<std::initializer_list<T>> list) {
        rows = list.size();
        if (rows == 0) {
            throw std::runtime_error("空矩阵初始化");
        }
        cols = list.begin()->size();
        data.resize(rows);
        size_t i = 0;
        for (const auto& row : list) {
            if (row.size() != cols) {
                throw std::runtime_error("矩阵初始化错误：行长度不一致");
            }
            data[i] = std::vector<T>(row);
            ++i;
        }
    }

    /**
     * @brief 拷贝构造函数
     * @param other 要拷贝的矩阵
     */
    Matrix(const Matrix& other) 
        : data(other.data), rows(other.rows), cols(other.cols) {}

    /**
     * @brief 赋值运算符
     * @param other 要赋值的矩阵
     * @return 对this的引用
     */
    Matrix& operator=(const Matrix& other) {
        if (this != &other) {
            data = other.data;
            rows = other.rows;
            cols = other.cols;
        }
        return *this;
    }

    /**
     * @brief 访问矩阵元素
     * @param i 行索引
     * @param j 列索引
     * @return 对元素的引用
     */
    T& operator()(size_t i, size_t j) noexcept { return data[i][j]; }
    const T& operator()(size_t i, size_t j) const noexcept { return data[i][j]; }
    /**
     * @brief 获取矩阵行数
     * @return 行数
     */
    size_t get_rows() const noexcept { return rows; }
    /**
     * @brief 获取矩阵列数
     * @return 列数
     */
    size_t get_cols() const noexcept { return cols; }

    /**
     * @brief 矩阵加法
     * @param rhs 右操作数
     * @return 和矩阵
     * @throw std::runtime_error 维度不匹配时抛出
     */
    Matrix operator+(const Matrix& rhs) const {
        if (rows != rhs.rows || cols != rhs.cols) {
            throw std::runtime_error("矩阵维度不匹配");
        }
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(i, j) = data[i][j] + rhs(i, j);
            }
        }
        return result;
    }

    /**
     * @brief 矩阵减法
     * @param rhs 右操作数
     * @return 差矩阵
     * @throw std::runtime_error 维度不匹配时抛出
     */
    Matrix operator-(const Matrix& rhs) const {
        if (rows != rhs.rows || cols != rhs.cols) {
            throw std::runtime_error("矩阵维度不匹配");
        }
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(i, j) = data[i][j] - rhs(i, j);
            }
        }
        return result;
    }

    /**
     * @brief 矩阵点乘（Hadamard积）
     * @param rhs 右操作数
     * @return 点乘结果
     * @throw std::runtime_error 维度不匹配时抛出
     */
    T dot(const Matrix& rhs) const {
        if (rows != rhs.rows || cols != rhs.cols) {
            throw std::runtime_error("矩阵维度不匹配");
        }
        T result = T();
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result += data[i][j] * rhs(i, j);
            }
        }
        return result;
    }

    /**
     * @brief 矩阵乘法
     * @param rhs 右操作数
     * @return 乘积矩阵
     * @throw std::runtime_error 维度不匹配时抛出
     */
    Matrix operator*(const Matrix& rhs) const {
        if (cols != rhs.rows) {
            throw std::runtime_error("矩阵维度不匹配");
        }
        Matrix result(rows, rhs.cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < rhs.cols; ++j) {
                T sum = T();
                for (size_t k = 0; k < cols; ++k) {
                    sum += data[i][k] * rhs(k, j);
                }
                result(i, j) = sum;
            }
        }
        return result;
    }

    /**
     * @brief 计算矩阵转置
     * @return 转置矩阵
     */
    Matrix transpose() const {
        Matrix result(cols, rows);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(j, i) = data[i][j];
            }
        }
        return result;
    }

    /**
     * @brief 计算矩阵的逆
     * @return 逆矩阵
     * @throw std::runtime_error 当矩阵不是方阵或不可逆时抛出
     * 
     * 使用高斯-约旦消元法计算矩阵的逆。
     * 如果矩阵接近奇异（行列式接近0），将抛出异常。
     */
    Matrix inverse() const {
        if (rows != cols) {
            throw std::runtime_error("非方阵无法求逆");
        }
        
        // 创建增广矩阵 [A|I]
        Matrix augmented(rows, cols * 2);
        
        // 填充增广矩阵的左半部分为原矩阵
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                augmented(i, j) = data[i][j];
            }
        }
        
        // 填充增广矩阵的右半部分为单位矩阵
        for (size_t i = 0; i < rows; ++i) {
            augmented(i, i + cols) = T(1);
        }
        
        // 高斯-约旦消元
        for (size_t i = 0; i < rows; ++i) {
            // 查找主元
            T pivot = augmented(i, i);
            if (std::abs(pivot) < T(1e-10)) {
                throw std::runtime_error("矩阵不可逆");
            }
            
            // 将主对角线元素归一化
            for (size_t j = i; j < cols * 2; ++j) {
                augmented(i, j) /= pivot;
            }
            
            // 消元
            for (size_t k = 0; k < rows; ++k) {
                if (k != i) {
                    T factor = augmented(k, i);
                    for (size_t j = i; j < cols * 2; ++j) {
                        augmented(k, j) -= factor * augmented(i, j);
                    }
                }
            }
        }
        
        // 提取结果矩阵（右半部分）
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(i, j) = augmented(i, j + cols);
            }
        }
        
        return result;
    }

    /**
     * @brief 计算矩阵的行列式
     * @return 行列式值
     * @throw std::runtime_error 非方阵时抛出
     * 
     * 使用递归方法计算行列式。
     * 对于2x2矩阵直接计算，
     * 对于更大的矩阵使用第一行展开。
     */
    T determinant() const {
        if (rows != cols) {
            throw std::runtime_error("非方阵无法计算行列式");
        }
        
        if (rows == 1) {
            return data[0][0];
        }
        
        if (rows == 2) {
            return data[0][0] * data[1][1] - data[0][1] * data[1][0];
        }
        
        T det = T();
        for (size_t j = 0; j < cols; ++j) {
            Matrix submatrix(rows - 1, cols - 1);
            for (size_t i = 1; i < rows; ++i) {
                for (size_t k = 0; k < cols; ++k) {
                    if (k < j) {
                        submatrix(i-1, k) = data[i][k];
                    } else if (k > j) {
                        submatrix(i-1, k-1) = data[i][k];
                    }
                }
            }
            det += (j % 2 == 0 ? 1 : -1) * data[0][j] * submatrix.determinant();
        }
        return det;
    }
};

} // namespace linalg
