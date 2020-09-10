#include "image_count8.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <iostream>
#include <vector>

using std::max;
using std::min;
using std::sqrt;
using std::vector;

double val_ratio = 0.5;

// 8方向，值不小于当前点时，累计数目，最后归一化：除以总数
int tran_func8(const unsigned char* s, float* t, const unsigned rows,
    const unsigned cols)
{
  // (1,0) up : 0 1 2 3 4 5 6 7
  typedef const unsigned char ST[rows][cols];
  auto& si = *(ST*)s;
  const unsigned len = 4 * (rows + cols),
                 dl[8] = { cols, rows, rows, cols, cols, rows, rows, cols };
  vector<unsigned> cnt(len, 0), cntn(len, 0);
  unsigned *p[8], *pn[8];
  float* res[8];
  p[0] = &cnt[0], pn[0] = &cntn[0];
  res[0] = t;
  for (unsigned d = 0; d < 7; ++d) {
    p[d + 1] = p[d] + dl[d];
    pn[d + 1] = pn[d] + dl[d];
    res[d + 1] = res[d] + dl[d];
  }
  for (unsigned r = 0; r < rows; ++r) {
    for (unsigned c = 0; c < cols; ++c) {
      if (si[r][c] == 0)
        continue;
      for (unsigned i = 0; i < rows; ++i) {
        for (unsigned j = 0; j < cols; ++j) {
          if ((i == r && j == c))
            continue;
          int dx = (int)i - r, dy = (int)j - c, l = max(abs(dx), abs(dy)), d;
          if (dx == 0) {
            d = dy > 0 ? 2 : 6;
          } else if (dy == 0) {
            d = dx > 0 ? 0 : 4;
          } else if (dx == dy) {
            d = dx > 0 ? 1 : 5;
          } else if (dx == -dy) {
            d = dx > 0 ? 7 : 3;
          } else {
            if (dx > 0) {
              if (dy > 0) {
                d = dx > dy ? 0 : 1;
              } else {
                d = dx > -dy ? 7 : 6;
              }
            } else {
              if (dy > 0) {
                d = -dx > dy ? 3 : 2;
              } else {
                d = -dx > -dy ? 4 : 5;
              }
            }
          }
          ++pn[d][l];
          if (si[i][j] >= si[r][c])
            ++p[d][l];
        }
      }
    }
  }
  for (unsigned d = 0; d < 8; ++d) {
    for (unsigned l = 1; l < dl[d]; ++l) {
      pn[d][0] += pn[d][l];
      p[d][0] += p[d][l];
    }
  }
  for (unsigned d = 0; d < 8; ++d) {
    for (unsigned l = 0; l < dl[d]; ++l) {
      res[d][l] = pn[d][l] > 0 ? (double)p[d][l] / pn[d][l] : 0;
    }
  }
  return 0;
}

// 8方向，累计平均值，最后除以总数
int tran_func80(const unsigned char* s, float* t, const unsigned rows,
    const unsigned cols)
{
  // (1,0) up : 0 1 2 3 4 5 6 7
  typedef const unsigned char ST[rows][cols];
  auto& si = *(ST*)s;
  const unsigned len = 4 * (rows + cols),
                 dl[8] = { cols, rows, rows, cols, cols, rows, rows, cols };
  vector<unsigned> sum(len, 0), sumn(len, 0);
  unsigned *p[8], *pn[8];
  float* res[8];
  p[0] = &sum[0], pn[0] = &sumn[0];
  res[0] = t;
  for (unsigned d = 0; d < 7; ++d) {
    p[d + 1] = p[d] + dl[d];
    pn[d + 1] = pn[d] + dl[d];
    res[d + 1] = res[d] + dl[d];
  }
  for (unsigned r = 0; r < rows; ++r) {
    for (unsigned c = 0; c < cols; ++c) {
      if (si[r][c] == 0)
        continue;
      for (unsigned i = 0; i < rows; ++i) {
        for (unsigned j = 0; j < cols; ++j) {
          if (i == r && j == c)
            continue;
          int dx = (int)i - r, dy = (int)j - c, l = max(abs(dx), abs(dy)), d;
          if (dx == 0) {
            d = dy > 0 ? 2 : 6;
          } else if (dy == 0) {
            d = dx > 0 ? 0 : 4;
          } else if (dx == dy) {
            d = dx > 0 ? 1 : 5;
          } else if (dx == -dy) {
            d = dx > 0 ? 7 : 3;
          } else {
            if (dx > 0) {
              if (dy > 0) {
                d = dx > dy ? 0 : 1;
              } else {
                d = dx > -dy ? 7 : 6;
              }
            } else {
              if (dy > 0) {
                d = -dx > dy ? 3 : 2;
              } else {
                d = -dx > -dy ? 4 : 5;
              }
            }
          }
          ++pn[d][l];
          p[d][l] += si[i][j];
        }
      }
    }
  }
  for (unsigned d = 0; d < 8; ++d) {
    for (unsigned l = 1; l < dl[d]; ++l) {
      pn[d][0] += pn[d][l];
      p[d][0] += p[d][l];
    }
  }
  float* t2 = t;
  for (unsigned d = 0; d < 8; ++d) {
    for (unsigned l = 0; l < dl[d]; ++l) {
      t2[l] = pn[d][l] > 0 ? (double)p[d][l] / pn[d][l] : 0;
    }
    t2 += dl[d];
  }
  return 0;
}

// 8方向，累计平均值，最后归一化：除以总数、最大值
int tran_func80n(const unsigned char* s, float* t, const unsigned rows,
    const unsigned cols)
{
  // (1,0) up : 0 1 2 3 4 5 6 7
  typedef const unsigned char ST[rows][cols];
  auto& si = *(ST*)s;
  const unsigned len = 4 * (rows + cols),
                 dl[8] = { cols, rows, rows, cols, cols, rows, rows, cols };
  vector<unsigned> sum(len, 0), sumn(len, 0);
  unsigned *p[8], *pn[8];
  float* res[8];
  p[0] = &sum[0], pn[0] = &sumn[0];
  res[0] = t;
  for (unsigned d = 0; d < 7; ++d) {
    p[d + 1] = p[d] + dl[d];
    pn[d + 1] = pn[d] + dl[d];
    res[d + 1] = res[d] + dl[d];
  }
  unsigned char mi = UCHAR_MAX, ma = 0;
  for (unsigned r = 0; r < rows; ++r) {
    for (unsigned c = 0; c < cols; ++c) {
      if (si[r][c] == 0)
        continue;
      mi = min(mi, si[r][c]);
      ma = max(ma, si[r][c]);
      for (unsigned i = 0; i < rows; ++i) {
        for (unsigned j = 0; j < cols; ++j) {
          if (i == r && j == c)
            continue;
          int dx = (int)i - r, dy = (int)j - c, l = max(abs(dx), abs(dy)), d;
          if (dx == 0) {
            d = dy > 0 ? 2 : 6;
          } else if (dy == 0) {
            d = dx > 0 ? 0 : 4;
          } else if (dx == dy) {
            d = dx > 0 ? 1 : 5;
          } else if (dx == -dy) {
            d = dx > 0 ? 7 : 3;
          } else {
            if (dx > 0) {
              if (dy > 0) {
                d = dx > dy ? 0 : 1;
              } else {
                d = dx > -dy ? 7 : 6;
              }
            } else {
              if (dy > 0) {
                d = -dx > dy ? 3 : 2;
              } else {
                d = -dx > -dy ? 4 : 5;
              }
            }
          }
          ++pn[d][l];
          p[d][l] += si[i][j];
        }
      }
    }
  }
  for (unsigned d = 0; d < 8; ++d) {
    for (unsigned l = 1; l < dl[d]; ++l) {
      pn[d][0] += pn[d][l];
      p[d][0] += p[d][l];
    }
  }
  float* t2 = t;
  for (unsigned d = 0; d < 8; ++d) {
    for (unsigned l = 0; l < dl[d]; ++l) {
      t2[l] = pn[d][l] > 0 ? (double)p[d][l] / pn[d][l] / ma : 0;
    }
    t2 += dl[d];
  }
  return 0;
}

// 8方向，累计非0平均值，最后归一化：除以总数、最大值
int tran_func80n2(const unsigned char* s, float* t, const unsigned rows,
    const unsigned cols)
{
  // (1,0) up : 0 1 2 3 4 5 6 7
  typedef const unsigned char ST[rows][cols];
  auto& si = *(ST*)s;
  const unsigned len = 4 * (rows + cols),
                 dl[8] = { cols, rows, rows, cols, cols, rows, rows, cols };
  vector<unsigned> sum(len, 0), sumn(len, 0);
  unsigned *p[8], *pn[8];
  float* res[8];
  p[0] = &sum[0], pn[0] = &sumn[0];
  res[0] = t;
  for (unsigned d = 0; d < 7; ++d) {
    p[d + 1] = p[d] + dl[d];
    pn[d + 1] = pn[d] + dl[d];
    res[d + 1] = res[d] + dl[d];
  }
  unsigned char mi = UCHAR_MAX, ma = 0;
  for (unsigned r = 0; r < rows; ++r) {
    for (unsigned c = 0; c < cols; ++c) {
      if (si[r][c] == 0)
        continue;
      mi = min(mi, si[r][c]);
      ma = max(ma, si[r][c]);
      for (unsigned i = 0; i < rows; ++i) {
        for (unsigned j = 0; j < cols; ++j) {
          if (si[i][j] == 0 || (i == r && j == c))
            continue;
          int dx = (int)i - r, dy = (int)j - c, l = max(abs(dx), abs(dy)), d;
          if (dx == 0) {
            d = dy > 0 ? 2 : 6;
          } else if (dy == 0) {
            d = dx > 0 ? 0 : 4;
          } else if (dx == dy) {
            d = dx > 0 ? 1 : 5;
          } else if (dx == -dy) {
            d = dx > 0 ? 7 : 3;
          } else {
            if (dx > 0) {
              if (dy > 0) {
                d = dx > dy ? 0 : 1;
              } else {
                d = dx > -dy ? 7 : 6;
              }
            } else {
              if (dy > 0) {
                d = -dx > dy ? 3 : 2;
              } else {
                d = -dx > -dy ? 4 : 5;
              }
            }
          }
          ++pn[d][l];
          p[d][l] += si[i][j];
        }
      }
    }
  }
  for (unsigned d = 0; d < 8; ++d) {
    for (unsigned l = 1; l < dl[d]; ++l) {
      pn[d][0] += pn[d][l];
      p[d][0] += p[d][l];
    }
  }
  float* t2 = t;
  for (unsigned d = 0; d < 8; ++d) {
    for (unsigned l = 0; l < dl[d]; ++l) {
      t2[l] = pn[d][l] > 0 ? (double)p[d][l] / pn[d][l] / ma : 0;
    }
    t2 += dl[d];
  }
  return 0;
}

// 8方向，累计所有平均值，最后归一化：除以总数、最大值
int tran_func80n3(const unsigned char* s, float* t, const unsigned rows,
    const unsigned cols)
{
  // (1,0) up : 0 1 2 3 4 5 6 7
  typedef const unsigned char ST[rows][cols];
  auto& si = *(ST*)s;
  const unsigned len = 4 * (rows + cols),
                 dl[8] = { cols, rows, rows, cols, cols, rows, rows, cols };
  vector<unsigned> sum(len, 0), sumn(len, 0);
  unsigned *p[8], *pn[8];
  float* res[8];
  p[0] = &sum[0], pn[0] = &sumn[0];
  res[0] = t;
  for (unsigned d = 0; d < 7; ++d) {
    p[d + 1] = p[d] + dl[d];
    pn[d + 1] = pn[d] + dl[d];
    res[d + 1] = res[d] + dl[d];
  }
  unsigned char mi = UCHAR_MAX, ma = 0;
  for (unsigned r = 0; r < rows; ++r) {
    for (unsigned c = 0; c < cols; ++c) {
      if (si[r][c] != 0) {
        mi = min(mi, si[r][c]);
        ma = max(ma, si[r][c]);
      }
      for (unsigned i = 0; i < rows; ++i) {
        for (unsigned j = 0; j < cols; ++j) {
          if (i == r && j == c)
            continue;
          int dx = (int)i - r, dy = (int)j - c, l = max(abs(dx), abs(dy)), d;
          if (dx == 0) {
            d = dy > 0 ? 2 : 6;
          } else if (dy == 0) {
            d = dx > 0 ? 0 : 4;
          } else if (dx == dy) {
            d = dx > 0 ? 1 : 5;
          } else if (dx == -dy) {
            d = dx > 0 ? 7 : 3;
          } else {
            if (dx > 0) {
              if (dy > 0) {
                d = dx > dy ? 0 : 1;
              } else {
                d = dx > -dy ? 7 : 6;
              }
            } else {
              if (dy > 0) {
                d = -dx > dy ? 3 : 2;
              } else {
                d = -dx > -dy ? 4 : 5;
              }
            }
          }
          ++pn[d][l];
          p[d][l] += si[i][j];
        }
      }
    }
  }
  for (unsigned d = 0; d < 8; ++d) {
    for (unsigned l = 1; l < dl[d]; ++l) {
      pn[d][0] += pn[d][l];
      p[d][0] += p[d][l];
    }
  }
  float* t2 = t;
  for (unsigned d = 0; d < 8; ++d) {
    for (unsigned l = 0; l < dl[d]; ++l) {
      t2[l] = pn[d][l] > 0 ? (double)p[d][l] / pn[d][l] / ma : 0;
    }
    t2 += dl[d];
  }
  return 0;
}

// 8方向，累计超过阈值点的平均值，最后归一化：除以总数、最大值
int tran_func80n4(const unsigned char* s, float* t, const unsigned rows,
    const unsigned cols)
{
  // (1,0) up : 0 1 2 3 4 5 6 7
  typedef const unsigned char ST[rows][cols];
  auto& si = *(ST*)s;
  const unsigned len = 4 * (rows + cols),
                 dl[8] = { cols, rows, rows, cols, cols, rows, rows, cols };
  vector<unsigned> sum(len, 0), sumn(len, 0);
  unsigned *p[8], *pn[8];
  float* res[8];
  p[0] = &sum[0], pn[0] = &sumn[0];
  res[0] = t;
  for (unsigned d = 0; d < 7; ++d) {
    p[d + 1] = p[d] + dl[d];
    pn[d + 1] = pn[d] + dl[d];
    res[d + 1] = res[d] + dl[d];
  }
  unsigned char mi = UCHAR_MAX, ma = 0;
  for (unsigned r = 0; r < rows; ++r) {
    for (unsigned c = 0; c < cols; ++c) {
      if (si[r][c] != 0) {
        mi = min(mi, si[r][c]);
        ma = max(ma, si[r][c]);
      }
    }
  }
  const unsigned char val = round(val_ratio * ma);
  for (unsigned r = 0; r < rows; ++r) {
    for (unsigned c = 0; c < cols; ++c) {
      if (si[r][c] < val)
        continue;
      for (unsigned i = 0; i < rows; ++i) {
        for (unsigned j = 0; j < cols; ++j) {
          if (i == r && j == c)
            continue;
          int dx = (int)i - r, dy = (int)j - c, l = max(abs(dx), abs(dy)), d;
          if (dx == 0) {
            d = dy > 0 ? 2 : 6;
          } else if (dy == 0) {
            d = dx > 0 ? 0 : 4;
          } else if (dx == dy) {
            d = dx > 0 ? 1 : 5;
          } else if (dx == -dy) {
            d = dx > 0 ? 7 : 3;
          } else {
            if (dx > 0) {
              if (dy > 0) {
                d = dx > dy ? 0 : 1;
              } else {
                d = dx > -dy ? 7 : 6;
              }
            } else {
              if (dy > 0) {
                d = -dx > dy ? 3 : 2;
              } else {
                d = -dx > -dy ? 4 : 5;
              }
            }
          }
          ++pn[d][l];
          p[d][l] += si[i][j];
        }
      }
    }
  }
  for (unsigned d = 0; d < 8; ++d) {
    for (unsigned l = 1; l < dl[d]; ++l) {
      pn[d][0] += pn[d][l];
      p[d][0] += p[d][l];
    }
  }
  float* t2 = t;
  for (unsigned d = 0; d < 8; ++d) {
    for (unsigned l = 0; l < dl[d]; ++l) {
      t2[l] = pn[d][l] > 0 ? (double)p[d][l] / pn[d][l] / ma : 0;
    }
    t2 += dl[d];
  }
  return 0;
}

// 8方向，值不小于当前点时，累计平均值，最后除以总数
int tran_func801(const unsigned char* s, float* t, const unsigned rows,
    const unsigned cols)
{
  // (1,0) up : 0 1 2 3 4 5 6 7
  typedef const unsigned char ST[rows][cols];
  auto& si = *(ST*)s;
  const unsigned len = 4 * (rows + cols),
                 dl[8] = { cols, rows, rows, cols, cols, rows, rows, cols };
  vector<unsigned> sum(len, 0), sumn(len, 0);
  unsigned *p[8], *pn[8];
  float* res[8];
  p[0] = &sum[0], pn[0] = &sumn[0];
  res[0] = t;
  for (unsigned d = 0; d < 7; ++d) {
    p[d + 1] = p[d] + dl[d];
    pn[d + 1] = pn[d] + dl[d];
    res[d + 1] = res[d] + dl[d];
  }
  for (unsigned r = 0; r < rows; ++r) {
    for (unsigned c = 0; c < cols; ++c) {
      if (si[r][c] == 0)
        continue;
      for (unsigned i = 0; i < rows; ++i) {
        for (unsigned j = 0; j < cols; ++j) {
          if (si[i][j] < si[r][c] || (i == r && j == c))
            continue;
          int dx = (int)i - r, dy = (int)j - c, l = max(abs(dx), abs(dy)), d;
          if (dx == 0) {
            d = dy > 0 ? 2 : 6;
          } else if (dy == 0) {
            d = dx > 0 ? 0 : 4;
          } else if (dx == dy) {
            d = dx > 0 ? 1 : 5;
          } else if (dx == -dy) {
            d = dx > 0 ? 7 : 3;
          } else {
            if (dx > 0) {
              if (dy > 0) {
                d = dx > dy ? 0 : 1;
              } else {
                d = dx > -dy ? 7 : 6;
              }
            } else {
              if (dy > 0) {
                d = -dx > dy ? 3 : 2;
              } else {
                d = -dx > -dy ? 4 : 5;
              }
            }
          }
          ++pn[d][l];
          p[d][l] += si[i][j];
        }
      }
    }
  }
  for (unsigned d = 0; d < 8; ++d) {
    for (unsigned l = 1; l < dl[d]; ++l) {
      pn[d][0] += pn[d][l];
      p[d][0] += p[d][l];
    }
  }
  float* t2 = t;
  for (unsigned d = 0; d < 8; ++d) {
    for (unsigned l = 0; l < dl[d]; ++l) {
      t2[l] = pn[d][l] > 0 ? (double)p[d][l] / pn[d][l] : 0;
    }
    t2 += dl[d];
  }
  return 0;
}

// 8方向，值不小于当前点时，累计平均值，最后归一化：除以总数、最大值
int tran_func801n(const unsigned char* s, float* t, const unsigned rows,
    const unsigned cols)
{
  // (1,0) up : 0 1 2 3 4 5 6 7
  typedef const unsigned char ST[rows][cols];
  auto& si = *(ST*)s;
  const unsigned len = 4 * (rows + cols),
                 dl[8] = { cols, rows, rows, cols, cols, rows, rows, cols };
  vector<unsigned> sum(len, 0), sumn(len, 0);
  unsigned *p[8], *pn[8];
  float* res[8];
  p[0] = &sum[0], pn[0] = &sumn[0];
  res[0] = t;
  for (unsigned d = 0; d < 7; ++d) {
    p[d + 1] = p[d] + dl[d];
    pn[d + 1] = pn[d] + dl[d];
    res[d + 1] = res[d] + dl[d];
  }
  unsigned char mi = UCHAR_MAX, ma = 0;
  for (unsigned r = 0; r < rows; ++r) {
    for (unsigned c = 0; c < cols; ++c) {
      if (si[r][c] == 0)
        continue;
      mi = min(mi, si[r][c]);
      ma = max(ma, si[r][c]);
      for (unsigned i = 0; i < rows; ++i) {
        for (unsigned j = 0; j < cols; ++j) {
          if (si[i][j] < si[r][c] || (i == r && j == c))
            continue;
          int dx = (int)i - r, dy = (int)j - c, l = max(abs(dx), abs(dy)), d;
          if (dx == 0) {
            d = dy > 0 ? 2 : 6;
          } else if (dy == 0) {
            d = dx > 0 ? 0 : 4;
          } else if (dx == dy) {
            d = dx > 0 ? 1 : 5;
          } else if (dx == -dy) {
            d = dx > 0 ? 7 : 3;
          } else {
            if (dx > 0) {
              if (dy > 0) {
                d = dx > dy ? 0 : 1;
              } else {
                d = dx > -dy ? 7 : 6;
              }
            } else {
              if (dy > 0) {
                d = -dx > dy ? 3 : 2;
              } else {
                d = -dx > -dy ? 4 : 5;
              }
            }
          }
          ++pn[d][l];
          p[d][l] += si[i][j];
        }
      }
    }
  }
  for (unsigned d = 0; d < 8; ++d) {
    for (unsigned l = 1; l < dl[d]; ++l) {
      pn[d][0] += pn[d][l];
      p[d][0] += p[d][l];
    }
  }
  float* t2 = t;
  for (unsigned d = 0; d < 8; ++d) {
    for (unsigned l = 0; l < dl[d]; ++l) {
      t2[l] = pn[d][l] > 0 ? (double)p[d][l] / pn[d][l] / ma : 0;
    }
    t2 += dl[d];
  }
  return 0;
}
// 8方向，累计平方平均值，最后除以总数
int tran_func82(const unsigned char* s, float* t, const unsigned rows,
    const unsigned cols)
{
  // (1,0) up : 0 1 2 3 4 5 6 7
  typedef const unsigned char ST[rows][cols];
  auto& si = *(ST*)s;
  const unsigned len = 4 * (rows + cols),
                 dl[8] = { cols, rows, rows, cols, cols, rows, rows, cols };
  vector<unsigned> sum(len, 0), sumn(len, 0);
  unsigned *p[8], *pn[8];
  p[0] = &sum[0], pn[0] = &sumn[0];
  for (unsigned i = 0; i < 7; ++i) {
    const unsigned l = i % 4 == 0 || i % 4 == 3 ? cols : rows;
    p[i + 1] = p[i] + l;
    pn[i + 1] = pn[i] + l;
  }
  for (unsigned r = 0; r < rows; ++r) {
    for (unsigned c = 0; c < cols; ++c) {
      if (si[r][c] == 0)
        continue;
      for (unsigned i = 0; i < rows; ++i) {
        for (unsigned j = 0; j < cols; ++j) {
          if (i == r && j == c)
            continue;
          int dx = (int)i - r, dy = (int)j - c, l = max(abs(dx), abs(dy)), d;
          if (dx == 0) {
            d = dy > 0 ? 2 : 6;
          } else if (dy == 0) {
            d = dx > 0 ? 0 : 4;
          } else if (dx == dy) {
            d = dx > 0 ? 1 : 5;
          } else if (dx == -dy) {
            d = dx > 0 ? 7 : 3;
          } else {
            if (dx > 0) {
              if (dy > 0) {
                d = dx > dy ? 0 : 1;
              } else {
                d = dx > -dy ? 7 : 6;
              }
            } else {
              if (dy > 0) {
                d = -dx > dy ? 3 : 2;
              } else {
                d = -dx > -dy ? 4 : 5;
              }
            }
          }
          ++pn[d][l];
          p[d][l] += si[i][j] * si[i][j];
        }
      }
    }
  }
  float* t2 = t;
  for (unsigned d = 0; d < 8; ++d) {
    for (unsigned l = 1; l < dl[d]; ++l) {
      pn[d][0] += pn[d][l];
      p[d][0] += p[d][l];
    }
  }
  for (unsigned d = 0; d < 8; ++d) {
    for (unsigned l = 0; l < dl[d]; ++l) {
      t2[l] = pn[d][l] > 0 ? sqrt((double)p[d][l] / pn[d][l]) : 0;
    }
    t2 += dl[d];
  }
  return 0;
}

// 8方向，累计平方平均值，最后归一化：除以总数、最大值
int tran_func82n(const unsigned char* s, float* t, const unsigned rows,
    const unsigned cols)
{
  // (1,0) up : 0 1 2 3 4 5 6 7
  typedef const unsigned char ST[rows][cols];
  auto& si = *(ST*)s;
  const unsigned len = 4 * (rows + cols),
                 dl[8] = { cols, rows, rows, cols, cols, rows, rows, cols };
  vector<unsigned> sum(len, 0), sumn(len, 0);
  unsigned *p[8], *pn[8];
  float* res[8];
  p[0] = &sum[0], pn[0] = &sumn[0];
  res[0] = t;
  for (unsigned d = 0; d < 7; ++d) {
    p[d + 1] = p[d] + dl[d];
    pn[d + 1] = pn[d] + dl[d];
    res[d + 1] = res[d] + dl[d];
  }
  unsigned char mi = UCHAR_MAX, ma = 0;
  for (unsigned r = 0; r < rows; ++r) {
    for (unsigned c = 0; c < cols; ++c) {
      if (si[r][c] == 0)
        continue;
      mi = min(mi, si[r][c]);
      ma = max(ma, si[r][c]);
      for (unsigned i = 0; i < rows; ++i) {
        for (unsigned j = 0; j < cols; ++j) {
          if (i == r && j == c)
            continue;
          int dx = (int)i - r, dy = (int)j - c, l = max(abs(dx), abs(dy)), d;
          if (dx == 0) {
            d = dy > 0 ? 2 : 6;
          } else if (dy == 0) {
            d = dx > 0 ? 0 : 4;
          } else if (dx == dy) {
            d = dx > 0 ? 1 : 5;
          } else if (dx == -dy) {
            d = dx > 0 ? 7 : 3;
          } else {
            if (dx > 0) {
              if (dy > 0) {
                d = dx > dy ? 0 : 1;
              } else {
                d = dx > -dy ? 7 : 6;
              }
            } else {
              if (dy > 0) {
                d = -dx > dy ? 3 : 2;
              } else {
                d = -dx > -dy ? 4 : 5;
              }
            }
          }
          ++pn[d][l];
          p[d][l] += si[i][j] * si[i][j];
        }
      }
    }
  }
  for (unsigned d = 0; d < 8; ++d) {
    for (unsigned l = 1; l < dl[d]; ++l) {
      pn[d][0] += pn[d][l];
      p[d][0] += p[d][l];
    }
  }
  float* t2 = t;
  for (unsigned d = 0; d < 8; ++d) {
    for (unsigned l = 0; l < dl[d]; ++l) {
      t2[l] = pn[d][l] > 0 ? (double)p[d][l] / pn[d][l] / ma : 0;
    }
    t2 += dl[d];
  }
  return 0;
}
