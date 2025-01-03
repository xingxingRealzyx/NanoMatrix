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
#include <sstream>

namespace linalg {

// 前向声明
template<typename T> class Matrix;

/**
 * @brief 向量类（作为矩阵的特例）
 * @tparam T 数值类型
 */
template<typename T>
class Vector : public Matrix<T> {
public:
    /** @brief 默认构造函数 */
    Vector() : Matrix<T>(0, 1) {}
    
    /**
     * @brief 指定大小构造函数
     * @param size 向量维度
     */
    explicit Vector(size_t size) : Matrix<T>(size, 1) {}
    
    /**
     * @brief 初始化列表构造函数
     * @param list 初始化数据
     */
    Vector(std::initializer_list<T> list) 
        : Matrix<T>(list.size(), 1) {
        size_t i = 0;
        for (const auto& val : list) {
            (*this)[i++] = val;
        }
    }

    /**
     * @brief 从矩阵构造向量（必须是列向量）
     * @param mat 输入矩阵
     * @throw std::runtime_error 当输入矩阵不是列向量时抛出
     */
    explicit Vector(const Matrix<T>& mat) {
        if (mat.get_cols() != 1) {
            throw std::runtime_error("只能从列向量矩阵构造向量");
        }
        *this = Vector(mat.get_rows());
        for (size_t i = 0; i < mat.get_rows(); ++i) {
            (*this)[i] = mat(i,0);
        }
    }

    /**
     * @brief 访问向量元素
     * @param i 索引
     * @return 对元素的引用
     */
    T& operator[](size_t i) { return Matrix<T>::operator()(i, 0); }
    const T& operator[](size_t i) const { return Matrix<T>::operator()(i, 0); }
    
    /**
     * @brief 获取向量维度
     * @return 向量维度
     */
    size_t size() const { return Matrix<T>::get_rows(); }

    /**
     * @brief 计算向量的范数
     * @return 向量的2-范数
     */
    T norm() const {
        return std::sqrt(this->dot(*this));
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
     * @brief 使用单个索引访问矩阵元素（按列优先顺序）
     * @param idx 线性索引
     * @return 对元素的引用
     * @throw std::out_of_range 索引越界时抛出
     */
    T& operator[](size_t idx) {
        if (idx >= rows * cols) {
            throw std::out_of_range("矩阵索引越界");
        }
        size_t i = idx / cols;  // 行索引
        size_t j = idx % cols;  // 列索引
        return data[i][j];
    }

    /**
     * @brief 使用单个索引访问矩阵元素（常量版本）
     */
    const T& operator[](size_t idx) const {
        if (idx >= rows * cols) {
            throw std::out_of_range("矩阵索引越界");
        }
        size_t i = idx / cols;
        size_t j = idx % cols;
        return data[i][j];
    }

    /**
     * @brief 使用(i,j)访问矩阵元素
     * @param i 行索引
     * @param j 列索引
     * @return 对元素的引用
     * @throw std::out_of_range 索引越界时抛出
     */
    T& operator()(size_t i, size_t j) {
        if (i >= rows || j >= cols) {
            throw std::out_of_range("矩阵索引越界");
        }
        return data[i][j];
    }

    /**
     * @brief 使用(i,j)访问矩阵元素（常量版本）
     */
    const T& operator()(size_t i, size_t j) const {
        if (i >= rows || j >= cols) {
            throw std::out_of_range("矩阵索引越界");
        }
        return data[i][j];
    }

    /**
     * @brief 获取矩阵的标量值（仅适用于1x1矩阵）
     * @return 矩阵的标量值
     * @throw std::runtime_error 矩阵不是1x1时抛出
     */
    T scalar() const {
        if (rows != 1 || cols != 1) {
            throw std::runtime_error("只有1x1矩阵可以转换为标量");
        }
        return data[0][0];
    }

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

    /**
     * @brief 计算矩阵每列的最大值
     * @return 行向量，包含每列的最大值
     */
    Matrix max() const {
        if (rows == 0 || cols == 0) {
            throw std::runtime_error("空矩阵无法计算最大值");
        }
        Matrix result(1, cols);  // 创建1×n的行向量
        for (size_t j = 0; j < cols; ++j) {
            T max_val = data[0][j];
            for (size_t i = 1; i < rows; ++i) {
                if (data[i][j] > max_val) {
                    max_val = data[i][j];
                }
            }
            result(0, j) = max_val;
        }
        return result;
    }

    /**
     * @brief 计算矩阵元素的绝对值
     * @return 新矩阵，每个元素都是原矩阵对应元素的绝对值
     */
    Matrix abs() const {
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(i,j) = std::abs(data[i][j]);
            }
        }
        return result;
    }

    /**
     * @brief 计算两个矩阵对应元素的最大值
     * @param rhs 右操作数
     * @return 新矩阵，每个元素是两个矩阵对应位置元素的最大值
     * @throw std::runtime_error 维度不匹配时抛出
     */
    Matrix max(const Matrix& rhs) const {
        if (rows != rhs.rows || cols != rhs.cols) {
            throw std::runtime_error("矩阵维度不匹配");
        }
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(i,j) = std::max(data[i][j], rhs(i,j));
            }
        }
        return result;
    }

    /**
     * @brief 打印矩阵到输出流
     * @param os 输出流
     * @param precision 数值精度
     */
    void print(std::ostream& os = std::cout, int precision = 4) const {
        // 保存原始格式设置
        auto old_flags = os.flags();
        auto old_precision = os.precision();
        
        // 设置新的格式
        os.precision(precision);
        os.setf(std::ios::fixed);
        
        // 计算每列的最大宽度
        std::vector<size_t> col_widths(cols);
        for (size_t j = 0; j < cols; ++j) {
            col_widths[j] = 0;
            for (size_t i = 0; i < rows; ++i) {
                std::ostringstream ss;
                ss.precision(precision);
                ss.setf(std::ios::fixed);
                ss << data[i][j];
                col_widths[j] = std::max(col_widths[j], ss.str().length());
            }
        }
        
        // 打印矩阵
        for (size_t i = 0; i < rows; ++i) {
            // 打印左括号
            os << (i == 0 ? "⎡" : (i == rows - 1 ? "⎣" : "⎢"));
            
            // 打印数据
            for (size_t j = 0; j < cols; ++j) {
                std::ostringstream ss;
                ss.precision(precision);
                ss.setf(std::ios::fixed);
                ss << data[i][j];
                std::string num = ss.str();
                
                // 右对齐
                os << std::string(col_widths[j] - num.length() + 1, ' ') << num;
                
                // 在数字之间添加适当的空格
                if (j < cols - 1) os << " ";
            }
            
            // 打印右括号
            os << (i == 0 ? " ⎤" : (i == rows - 1 ? " ⎦" : " ⎥"));
            os << "\n";
        }
        
        // 恢复原始格式设置
        os.flags(old_flags);
        os.precision(old_precision);
    }

    /**
     * @brief 重载输出运算符
     */
    friend std::ostream& operator<<(std::ostream& os, const Matrix& mat) {
        mat.print(os);
        return os;
    }

    /**
     * @brief 矩阵与标量相乘
     * @param scalar 标量值
     * @return 乘积矩阵
     */
    Matrix operator*(T scalar) const {
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(i, j) = data[i][j] * scalar;
            }
        }
        return result;
    }
};

/**
 * @brief 计算矩阵每列的最大值
 * @param mat 输入矩阵
 * @return 行向量，包含每列的最大值
 */
template<typename T>
Matrix<T> max(const Matrix<T>& mat) {
    return mat.max();
}

/**
 * @brief 计算两个矩阵对应元素的最大值
 * @param lhs 左操作数
 * @param rhs 右操作数
 * @return 新矩阵，每个元素是两个矩阵对应位置元素的最大值
 */
template<typename T>
Matrix<T> max(const Matrix<T>& lhs, const Matrix<T>& rhs) {
    return lhs.max(rhs);
}

/**
 * @brief 计算矩阵元素的绝对值
 * @param mat 输入矩阵
 * @return 新矩阵，每个元素都是原矩阵对应元素的绝对值
 */
template<typename T>
Matrix<T> abs(const Matrix<T>& mat) {
    return mat.abs();
}

/**
 * @brief 标量与矩阵相乘（左乘）
 * @param scalar 标量值
 * @param mat 矩阵
 * @return 乘积矩阵
 */
template<typename T>
Matrix<T> operator*(T scalar, const Matrix<T>& mat) {
    return mat * scalar;  // 复用矩阵右乘标量的实现
}

} // namespace linalg
