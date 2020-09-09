#ifndef _MAIN_
#define _MAIN_

#include <bits/stdc++.h>
#ifdef _WIN32
#include <winsock2.h>
#else // _LINUX
#include <netinet/in.h>
#endif

using namespace std;

void tran8(unsigned char* source, float* target, const unsigned images);
void tran80(unsigned char* source, float* target, const unsigned images);

using out_t = float;
// auto& tran = tran8;
// string dir = "../data/MNIST_data/", outdir = "../data/MNIST_count8_data/";
auto& tran = tran80;
string dir = "../data/MNIST_data/", outdir = "../data/MNIST_count80_data/";
string filenames[2] = { "t10k-images-idx3-ubyte", "train-images-idx3-ubyte" };
const unsigned rows = 28, cols = 28, len = 4 * (rows + cols);

void tran_func8(const unsigned char si[rows][cols], float ti[len])
{
  // (1,0) up : 0 1 2 3 4 5 6 7
  unsigned cnt[8] = { 0 }, sumr[4][rows] = { { 0 } },
           sumc[4][cols] = { { 0 } }, sumnr[4][rows] = { { 0 } },
           sumnc[4][cols] = { { 0 } },
           *p[8] = { sumc[0], sumr[0], sumr[1], sumc[1], sumc[2], sumr[2],
             sumr[3], sumc[3] },
           *pn[8] = { sumnc[0], sumnr[0], sumnr[1], sumnc[1], sumnc[2],
             sumnr[2], sumnr[3], sumnc[3] };
  static constexpr unsigned dl[8]
      = { cols, rows, rows, cols, cols, rows, rows, cols };
  for (unsigned r = 0; r < rows; ++r) {
    for (unsigned c = 0; c < cols; ++c) {
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
          ++cnt[d];
          if (0 < si[r][c]) {
            ++pn[d][l];
            if (si[i][j] >= si[r][c])
              ++p[d][l];
          }
        }
      }
    }
  }
  float* t = ti;
  for (unsigned d = 0; d < 8; ++d) {
    for (unsigned l = 1; l < dl[d]; ++l) {
      pn[d][0] += pn[d][l];
      p[d][0] += p[d][l];
    }
  }
  for (unsigned d = 0; d < 8; ++d) {
    for (unsigned l = 0; l < dl[d]; ++l) {
      t[l] = pn[d][l] > 0 ? (double)p[d][l] / pn[d][l] : 0;
    }
    // t[0] = cnt[d] > 0 ? (double)pn[d][0] / cnt[d] : 0;
    t += dl[d];
  }
}

void tran_func8_back(unsigned char si[rows][cols], float ti[len])
{
  // (1,0) up : 1 3 2 5 4 6 7 0
  unsigned cnt[len] = { 0 }, cntn[len] = { 0 }, *p[8], *pn[8];
  float* res[8];
  p[0] = cnt, pn[0] = cntn;
  res[0] = ti;
  for (unsigned i = 0; i < 7; ++i) {
    p[i + 1] = p[i] + ((i / 2 % 2 == 0) ? cols : rows);
    pn[i + 1] = pn[i] + ((i / 2 % 2 == 0) ? cols : rows);
    res[i + 1] = res[i] + ((i / 2 % 2 == 0) ? cols : rows);
  }
  for (unsigned r = 0; r < rows; ++r) {
    for (unsigned c = 0; c < cols; ++c) {
      if (si[r][c] == 0)
        continue;
      for (unsigned i = 0; i < rows; ++i) {
        for (unsigned j = 0; j < cols; ++j) {
          if (i == r && j == c)
            continue;
          int dx = (int)i - r, dy = (int)j - c, l = max(abs(dx), abs(dy));
          bool f = si[i][j] >= si[r][c];
          if (dx == 0) {
            if (dy > 0) {
              ++pn[2][l];
              // ++pn[3][l];
              if (f) {
                ++p[2][l];
                // ++p[3][l];
              }
            } else {
              // ++pn[6][l];
              ++pn[7][l];
              if (f) {
                // ++p[6][l];
                ++p[7][l];
              }
            }
          } else if (dy == 0) {
            if (dx > 0) {
              // ++pn[0][l];
              ++pn[1][l];
              if (f) {
                // ++p[0][l];
                ++p[1][l];
              }
            } else {
              ++pn[4][l];
              // ++pn[5][l];
              if (f) {
                ++p[4][l];
                // ++p[5][l];
              }
            }
          } else if (dx == dy) {
            if (dx > 0) {
              // ++pn[1][l];
              ++pn[3][l];
              if (f) {
                // ++p[1][l];
                ++p[3][l];
              }
            } else {
              // ++pn[4][l];
              ++pn[6][l];
              if (f) {
                //++p[4][l];
                ++p[6][l];
              }
            }
          } else if (dx == -dy) {
            if (dx > 0) {
              ++pn[0][l];
              //++pn[7][l];
              if (f) {
                ++p[0][l];
                // ++p[7][l];
              }
            } else {
              // ++pn[2][l];
              ++pn[5][l];
              if (f) {
                // ++p[2][l];
                ++p[5][l];
              }
            }
          } else {
            unsigned d = dy + dx > 0 ? 0 : 6;
            if (dy > dx)
              d ^= 2;
            if (d / 2 % 2 == 0) {
              if (dy > 0)
                d |= 1;
            } else {
              if (dx > 0)
                d |= 1;
            }
            ++pn[d][l];
            if (f)
              ++p[d][l];
          }
        }
      }
    }
  }
  for (unsigned d = 0; d < 8; ++d) {
    for (unsigned l = 1, le = d / 2 % 2 == 0 ? cols : rows; l < le; ++l) {
      pn[d][0] += pn[d][l];
      p[d][0] += p[d][l];
    }
  }
  for (unsigned d = 0; d < 8; ++d) {
    for (unsigned l = 0, le = d / 2 % 2 == 0 ? cols : rows; l < le; ++l) {
      res[d][l] = pn[d][l] > 0 ? (double)p[d][l] / pn[d][l] : 0;
    }
  }
}

void tran8(unsigned char* source, float* target, const unsigned images)
{
  using dt = unsigned char[rows][cols];
  dt* s = (dt*)source;
  for (unsigned i = 0; i < images; ++i) {
    tran_func8(s[i], target + len * i);
  }
}

void tran_func80(const unsigned char si[rows][cols], float ti[len])
{
  // (1,0) up : 0 1 2 3 4 5 6 7
  unsigned cnt[8] = { 0 }, sumr[4][rows] = { { 0 } },
           sumc[4][cols] = { { 0 } }, sumnr[4][rows] = { { 0 } },
           sumnc[4][cols] = { { 0 } },
           *p[8] = { sumc[0], sumr[0], sumr[1], sumc[1], sumc[2], sumr[2],
             sumr[3], sumc[3] },
           *pn[8] = { sumnc[0], sumnr[0], sumnr[1], sumnc[1], sumnc[2],
             sumnr[2], sumnr[3], sumnc[3] };
  static constexpr unsigned dl[8]
      = { cols, rows, rows, cols, cols, rows, rows, cols };
  unsigned char mi = UCHAR_MAX, ma = 0;
  for (unsigned r = 0; r < rows; ++r) {
    for (unsigned c = 0; c < cols; ++c) {
      if (si[r][c] != 0) {
        mi = min(mi, si[r][c]);
        ma = max(ma, si[r][c]);
      }
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
          ++cnt[d];
          if (0 < si[r][c]) {
            ++pn[d][l];
            if (si[i][j] >= si[r][c])
              p[d][l] += si[i][j];
          }
        }
      }
    }
  }
  double rmax = ma == mi ? 1.0 : 1.0 / (ma - mi);
  float* t = ti;
  for (unsigned d = 0; d < 8; ++d) {
    for (unsigned l = 1; l < dl[d]; ++l) {
      pn[d][0] += pn[d][l];
      p[d][0] += p[d][l];
    }
  }
  for (unsigned d = 0; d < 8; ++d) {
    for (unsigned l = 0; l < dl[d]; ++l) {
      t[l] = pn[d][l] > 0 ? rmax * ((double)p[d][l] / pn[d][l] - mi) : 0;
    }
    // t[0] = cnt[d] > 0 ? (double)pn[d][0] / cnt[d] : 0;
    t += dl[d];
  }
}

void tran80(unsigned char* source, float* target, const unsigned images)
{
  using dt = unsigned char[rows][cols];
  dt* s = (dt*)source;
  for (unsigned i = 0; i < images; ++i) {
    tran_func80(s[i], target + len * i);
  }
}

int handle(string& filename)
{
  ifstream is;
  is.open(dir + filename, ios::binary | ios::in);
  if (!is) {
    cerr << "open file error: " << dir + filename << endl;
    return -1;
  }
  unsigned char buf[16];
  char* sbuf = (char*)buf;
  is.read(sbuf, 16);
  unsigned *ibuf = (unsigned*)buf, magic = ntohl(ibuf[0]), tt = buf[2],
           dim = buf[3], images = ntohl(ibuf[1]), rows = ntohl(ibuf[2]),
           cols = ntohl(ibuf[3]);
  cout << magic << '\t' << tt << '\t' << dim << '\t' << images << '\t' << rows
       << '\t' << cols << endl;
  unsigned bsize = images * rows * cols;
  vector<unsigned char> data(bsize);
  is.read((char*)&data[0], bsize);
  is.close();
  cout << "read file: " << dir + filename << endl;
  const unsigned outlen = images * len;
  vector<out_t> res(outlen, 0);
  tran(&data[0], &res[0], images);
  if (typeid(out_t) == typeid(unsigned char))
    buf[2] = 0x8;
  else if (typeid(res[0]) == typeid(char))
    buf[2] = 0x9;
  else if (typeid(res[0]) == typeid(int16_t)) {
    buf[2] = 0xB;
    for (unsigned i = 0; i < outlen; ++i) {
      res[i] = htons(res[i]);
    }
  } else if (typeid(res[0]) == typeid(int32_t)) {
    buf[2] = 0xC;
    for (unsigned i = 0; i < outlen; ++i) {
      res[i] = htonl(res[i]);
    }
  } else if (typeid(res[0]) == typeid(float))
    buf[2] = 0xD;
  else if (typeid(res[0]) == typeid(double))
    buf[2] = 0xE;
  else {
    cerr << typeid(res[0]).name() << endl;
    return -1;
  }
  buf[3] = 2; // dim
  ibuf[1] = htonl(images);
  ibuf[2] = htonl(len);
  ofstream os;
  os.open(outdir + filename, ios::binary | ios::out);
  if (!os) {
    cerr << "open file error: " << outdir + filename << endl;
    return -1;
  }
  os.write(sbuf, 12);
  os.write((char*)&res[0], outlen * sizeof(out_t));
  os.close();
  cout << "save to file: " << outdir + filename << endl;
  cout << ntohl(ibuf[0]) << "\t" << (unsigned)buf[2] << "\t"
       << (unsigned)buf[3] << "\t" << images << "\t" << len << endl;
  return 0;
}

//*****************************************************************//*
int main(int argc, char** argv)
{
  for (auto& filename : filenames) {
    handle(filename);
  }
  return 0;
}

//*****************************************************************//*
#endif // _MAIN_
