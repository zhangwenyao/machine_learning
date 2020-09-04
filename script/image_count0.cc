#ifndef _MAIN_
#define _MAIN_

#include <bits/stdc++.h>
#include <netinet/in.h>

using namespace std;

string dir = "../data/MNIST_data/", outdir = "../data/MNIST_count_data/";
string filenames[2] = { "t10k-images-idx3-ubyte", "train-images-idx3-ubyte" };
const int rows = 28, cols = 28;

void tran_func(unsigned char si[rows][cols], float* ti)
{ // (1,0) up : 1 3 2 5 4 6 7 0
  const unsigned size = 4 * (rows + cols);
  int cnt[size] = { 0 }, cntn[size] = { 0 }, *p[8], *pn[8];
  float* res[8];
  p[0] = cnt, pn[0] = cntn;
  res[0] = ti;
  for (int i = 0; i < 7; ++i) {
    p[i + 1] = p[i] + ((i / 2 % 2 == 0) ? cols : rows);
    pn[i + 1] = pn[i] + ((i / 2 % 2 == 0) ? cols : rows);
    res[i + 1] = res[i] + ((i / 2 % 2 == 0) ? cols : rows);
  }
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      if (si[r][c] == 0)
        continue;
      for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
          if (i == r && j == c)
            continue;
          int dx = i - r, dy = j - c, l = max(abs(dx), abs(dy));
          bool f = si[i][j] >= si[r][c];
          if (dx == 0) {
            if (dy > 0) {
              ++pn[2][l];
              ++pn[3][l];
              if (f) {
                ++p[2][l];
                ++p[3][l];
              }
            } else {
              ++pn[6][l];
              ++pn[7][l];
              if (f) {
                ++p[6][l];
                ++p[7][l];
              }
            }
          } else if (dy == 0) {
            if (dx > 0) {
              ++pn[0][l];
              ++pn[1][l];
              if (f) {
                ++p[0][l];
                ++p[1][l];
              }
            } else {
              ++pn[4][l];
              ++pn[5][l];
              if (f) {
                ++p[4][l];
                ++p[5][l];
              }
            }
          } else if (dx == dy) {
            if (dx > 0) {
              ++pn[1][l];
              ++pn[3][l];
              if (f) {
                ++p[1][l];
                ++p[3][l];
              }
            } else {
              ++pn[4][l];
              ++pn[6][l];
              if (f) {
                ++p[4][l];
                ++p[6][l];
              }
            }
          } else if (dx == -dy) {
            if (dx > 0) {
              ++pn[0][l];
              ++pn[7][l];
              if (f) {
                ++p[0][l];
                ++p[7][l];
              }
            } else {
              ++pn[2][l];
              ++pn[5][l];
              if (f) {
                ++p[2][l];
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
  for (int d = 0; d < 8; ++d) {
    for (int l = 1, le = d / 2 % 2 == 0 ? cols : rows; l < le; ++l) {
      pn[d][0] += pn[d][l];
      p[d][0] += p[d][l];
    }
  }
  for (int d = 0; d < 8; ++d) {
    for (int l = 1, le = d / 2 % 2 == 0 ? cols : rows; l < le; ++l) {
      res[d][l] = pn[d][l] > 0 ? (double)p[d][l] / pn[d][l] : 0;
    }
  }
}

void tran(unsigned char* source, float* target, const unsigned images)
{
  using dt = unsigned char[rows][cols];
  dt* s = (dt*)source;
  const unsigned size = 4 * (rows + cols);
  for (unsigned i = 0; i < images; ++i) {
    auto& si = s[i];
    float* ti = target + size * i;
    tran_func(si, ti);
  }
}

int handle(string& filename)
{
  ifstream is;
  is.open(dir + filename, ios::binary | ios::in);
  if (!is) {
    cout << "open file error: " << dir + filename << endl;
    return -1;
  }
  unsigned char buf[16];
  char* cbuf = (char*)buf;
  is.read(cbuf, 16);
  unsigned *ibuf = (unsigned*)buf, ds = buf[3], magic = ntohl(ibuf[0]),
           images = ntohl(ibuf[1]), rows = ntohl(ibuf[2]),
           cols = ntohl(ibuf[3]);
  cout << magic << '\t' << ds << '\t' << images << '\t' << rows << '\t'
       << cols << endl;
  unsigned size = images * rows * cols;
  vector<unsigned char> data(size);
  is.read((char*)&data[0], size);
  is.close();
  cout << "read file: " << dir + filename << endl;

  unsigned len = 4 * (rows + cols), outsize = images * len * sizeof(float);
  vector<float> res(len * images, 0);
  tran(&data[0], &res[0], images);
  ofstream os;
  os.open(outdir + filename, ios::binary | ios::out);
  if (!os) {
    cout << "open file error: " << outdir + filename << endl;
    return -1;
  }
  buf[2] = 0x0D; // float32
  buf[3] = 2;    // dim
  // buf[4-7] : images
  *(unsigned*)(buf + 8) = htonl(len);
  os.write(cbuf, 12);
  os.write((char*)&res[0], outsize);
  os.close();
  cout << "save to file: " << outdir + filename << endl;
  return 0;
}

// **********************************************************
int main(int argc, char** argv)
{
  for (auto& filename : filenames) {
    handle(filename);
  }
  return 0;
}

//*****************************************************************//*
#endif // _MAIN_
