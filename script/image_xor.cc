#ifndef _MAIN_
#define _MAIN_

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stack>
#include <string>
#include <utility>
#include <vector>

using namespace std;

string dir = "../data/MNIST_data/", outdir = "../data/MNIST_xor_mid_data/";
string filenames[2] = { "t10k-images-idx3-ubyte", "train-images-idx3-ubyte" };

inline void char2int(unsigned char* c, unsigned& i)
{
  i = ((((((unsigned)c[0] << 8) | c[1]) << 8) | c[2]) << 8) | c[3];
}

inline void int2char(unsigned i, unsigned char* c)
{
  c[0] = i & 0xff000000;
  c[1] = i & 0xff0000;
  c[2] = i & 0xff00;
  c[3] = i & 0xff;
}

void tran_row(unsigned char* target, const unsigned rows, const unsigned cols,
    unsigned t, unsigned s)
{
  using dt = char[cols];
  dt* ti = (dt*)target;
  for (unsigned c = 0; c < cols; ++c) {
    ti[t][c] ^= ti[s][c];
  }
}
void tran_col(unsigned char* target, const unsigned rows, const unsigned cols,
    unsigned t, unsigned s)
{
  using dt = char[cols];
  dt* ti = (dt*)target;
  for (unsigned r = 0; r < rows; ++r) {
    ti[r][t] ^= ti[r][s];
  }
}
using func_t = decltype(tran_row);
void tran_func(unsigned char* target, const unsigned rows,
    const unsigned cols, unsigned b, unsigned e, bool f, func_t* func)
{
  if (b >= e)
    return;
  unsigned len = e - b + 1;
  if (len % 2 == 1) {
    if (f) {
      tran_func(target, rows, cols, b + 1, e, f, func);
      func(target, rows, cols, b, b + 1);
    } else {
      tran_func(target, rows, cols, b, e - 1, f, func);
      func(target, rows, cols, e, e - 1);
    }
    return;
  }
  unsigned emid = b + len / 2, bmid = emid - 1;
  tran_func(target, rows, cols, b, bmid, f, func);
  tran_func(target, rows, cols, emid, e, f, func);
  if (f) {
    for (unsigned i = b; i <= bmid; ++i)
      func(target, rows, cols, i, i + len / 2);
  } else {
    for (unsigned i = b; i <= bmid; ++i)
      func(target, rows, cols, i + len / 2, i);
  }
}

void tran(unsigned char* source, unsigned char* target, const unsigned images,
    const unsigned rows, const unsigned cols)
{
  const unsigned size = images * rows * cols, rmid = rows / 2,
                 cmid = cols / 2;
  memcpy(target, source, size);
  if (rows <= 2 && cols <= 2)
    return;
  using dt = char[rows][cols];
  dt* t = (dt*)target;
  for (unsigned i = 0; i < images; ++i) {
    auto& ti = t[i];
    tran_func((unsigned char*)ti, rows, cols, 0, rmid, false, tran_row);
    tran_func((unsigned char*)ti, rows, cols, rmid, rows - 1, true, tran_row);
    tran_func((unsigned char*)ti, rows, cols, 0, cmid, false, tran_col);
    tran_func((unsigned char*)ti, rows, cols, cmid, cols - 1, true, tran_col);
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
  unsigned char ubuf[16];
  char* buf = (char*)ubuf;
  unsigned magic, ds, images, rows, cols;
  is.read(buf, 16);
  char2int(ubuf, magic);
  ds = magic & 0xff;
  char2int(ubuf + 4, images);
  char2int(ubuf + 8, rows);
  char2int(ubuf + 12, cols);
  cout << magic << '\t' << ds << '\t' << images << '\t' << rows << '\t'
       << cols << endl;
  unsigned size = images * rows * cols;
  vector<unsigned char> udata(size), ures(size);
  char *data = (char*)&udata[0], *res = (char*)&ures[0];
  is.read(data, size);
  is.close();
  cout << "read file: " << dir + filename << endl;
  tran(&udata[0], &ures[0], images, rows, cols);
  ofstream os;
  os.open(outdir + filename, ios::binary | ios::out);
  if (!os) {
    cout << "open file error: " << outdir + filename << endl;
    return -1;
  }
  os.write(buf, 16);
  os.write(&res[0], size);
  os.close();
  cout << "save to file: " << outdir + filename << endl;
  return 0;
}

// **********************************************************
int main(int argc, char** argv)
{
  for (auto& filename : filenames)
    handle(filename);
  return 0;
}

//*****************************************************************//*
#endif // _MAIN_
