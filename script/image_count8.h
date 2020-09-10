#ifndef IMAGE_COUNT8_H_
#define IMAGE_COUNT8_H_

extern double val_ratio;

template <typename T, typename T2>
int tran8(unsigned char* source, T* target, const unsigned images,
    const unsigned rows, const unsigned cols, T2 func);

// 8方向，值不小于当前点时，累计数目，最后归一化：除以总数
int tran_func8(const unsigned char* s, float* t, const unsigned rows,
    const unsigned cols);

// 8方向，累计平均值，最后除以总数
int tran_func80(const unsigned char* s, float* t, const unsigned rows,
    const unsigned cols);
// 8方向，累计平均值，最后归一化：除以总数、最大值
int tran_func80n(const unsigned char* s, float* t, const unsigned rows,
    const unsigned cols);
// 8方向，累计非0平均值，最后归一化：除以总数、最大值
int tran_func80n2(const unsigned char* s, float* t, const unsigned rows,
    const unsigned cols);
// 8方向，累计所有平均值，最后归一化：除以总数、最大值
int tran_func80n3(const unsigned char* s, float* t, const unsigned rows,
    const unsigned cols);
// 8方向，累计超过阈值点的平均值，最后归一化：除以总数、最大值
int tran_func80n4(const unsigned char* s, float* t, const unsigned rows,
    const unsigned cols);
// 8方向，值不小于当前点时，累计平均值，最后除以总数
int tran_func801(const unsigned char* s, float* t, const unsigned rows,
    const unsigned cols);
// 8方向，值不小于当前点时，累计平均值，最后归一化：除以总数、最大值
int tran_func801n(const unsigned char* s, float* t, const unsigned rows,
    const unsigned cols);

// 8方向，累计平方平均值，最后除以总数
int tran_func82(const unsigned char* s, float* t, const unsigned rows,
    const unsigned cols);
// 8方向，累计平方平均值，最后归一化：除以总数、最大值
int tran_func82n(const unsigned char* s, float* t, const unsigned rows,
    const unsigned cols);

#include "image_count8_template.h"
#endif
