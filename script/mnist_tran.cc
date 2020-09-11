#ifndef _MAIN_
#define _MAIN_

#include "image_count8.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <direct.h>
#include <io.h>
#include <winsock2.h>
#define ACCESS _access
#define MKDIR(a) _mkdir((a))
#else // _LINUX
#include <netinet/in.h>
#include <sys/stat.h>
#include <unistd.h>
#define ACCESS access
#define MKDIR(a) mkdir((a), 0755)
#endif

using std::cerr;
using std::cout;
using std::endl;
using std::ifstream;
using std::ios;
using std::ofstream;
using std::round;
using std::string;
using std::vector;

#define DIGITS_TXT2MNIST 0
#define DIGITS_COUNT 1
#define MNIST_COUNT 0

#if DIGITS_TXT2MNIST
using out_t = unsigned char;
static constexpr unsigned images = 1797, rows = 8, cols = 8;
string dir = "../data/digits_data/", outdir = "../data/digits_data/";
string filenames[] = { "data", "target" };
#elif DIGITS_COUNT
using out_t = float;
static constexpr unsigned rows = 8, cols = 8;
string dir = "../data/digits_data/", outdir = "../data/digits_count8_data/";
string filenamesx[] = { "data.dat" }, filenamesy[] = { "target.dat" };
#else // MNIST_COUNT
using out_t = float;
static constexpr unsigned rows = 28, cols = 28;
string dir = "../data/MNIST_data/", outdir = "../data/MNIST_count86_data/";
string filenamesx[] = { "t10k-images-idx3-ubyte", "train-images-idx3-ubyte" },
       filenamesy[] = { "t10k-labels-idx1-ubyte", "train-labels-idx1-ubyte" };
#endif

static constexpr unsigned len = 4 * (rows + cols);
auto& tran_func = tran_func8;
auto& tran = tran8<out_t, decltype(tran_func)>;

// static constexpr unsigned len = 4 * (rows + cols) - 7;
// auto& tran_func = tran_func870n;
// auto& tran = tran87<out_t, decltype(tran_func)>;

// static constexpr unsigned len = 4 * (rows + cols - 2);
// auto& tran_func = tran_func86;
// auto& tran = tran86<out_t, decltype(tran_func)>;

double ratio = 1.0;

int mnist_x(string& filename)
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
  images = round(images * ratio);
  const unsigned outlen = images * len;
  vector<float> res(outlen, 0);
  if (0 != tran(&data[0], &res[0], images, rows, cols, tran_func)) {
    return -1;
  }
  if (typeid(out_t) == typeid(unsigned char))
    buf[2] = 0x8;
  else if (typeid(out_t) == typeid(char))
    buf[2] = 0x9;
  else if (typeid(out_t) == typeid(int16_t)) {
    buf[2] = 0xB;
    for (unsigned i = 0; i < outlen; ++i) {
      res[i] = htons(res[i]);
    }
  } else if (typeid(out_t) == typeid(int32_t)) {
    buf[2] = 0xC;
    for (unsigned i = 0; i < outlen; ++i) {
      res[i] = htonl(res[i]);
    }
  } else if (typeid(out_t) == typeid(float))
    buf[2] = 0xD;
  else if (typeid(out_t) == typeid(double))
    buf[2] = 0xE;
  else {
    cerr << typeid(out_t).name() << endl;
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

int mnist_y(string& filename)
{
  ifstream is;
  is.open(dir + filename, ios::binary | ios::in);
  if (!is) {
    cerr << "open file error: " << dir + filename << endl;
    return -1;
  }
  unsigned char buf[8];
  char* sbuf = (char*)buf;
  is.read(sbuf, 8);
  unsigned *ibuf = (unsigned*)buf, magic = ntohl(ibuf[0]), tt = buf[2],
           dim = buf[3], images = ntohl(ibuf[1]);
  cout << magic << '\t' << tt << '\t' << dim << '\t' << images << endl;
  vector<unsigned char> data(images);
  is.read((char*)&data[0], images);
  is.close();
  cout << "read file: " << dir + filename << endl;
  images = round(images * ratio);
  // buf[3] : // dim
  ibuf[1] = htonl(images);
  ofstream os;
  os.open(outdir + filename, ios::binary | ios::out);
  if (!os) {
    cerr << "open file error: " << outdir + filename << endl;
    return -1;
  }
  os.write(sbuf, 8);
  os.write((char*)&data[0], images);
  os.close();
  cout << "save to file: " << outdir + filename << endl;
  cout << ntohl(ibuf[0]) << "\t" << (unsigned)buf[2] << "\t"
       << (unsigned)buf[3] << "\t" << images << endl;
  return 0;
}

int mkdirs(const char* dirname)
{
  int i = 0;
  int iRet;
  char dir[1000];
  int iLen = strlen(dirname);
  if (iLen <= 0 || iLen >= 998)
    return -1;
  strcpy(dir, dirname);
  //在末尾加/
  if (dir[iLen - 1] != '\\' && dir[iLen - 1] != '/') {
    dir[iLen] = '/';
    dir[++iLen] = '\0';
  }

  // 创建目录
  for (i = 0; i < iLen; i++) {
    if (dir[i] == '\\' || dir[i] == '/') {
      if (i == 0) {
        dir[i] = '/';
        continue;
      }

      dir[i] = '\0';

      //如果不存在,创建
      iRet = ACCESS(dir, 0);
      if (iRet != 0) {
        iRet = MKDIR(dir);
        if (iRet != 0) {
          cerr << dir << endl;
          return -1;
        }
      }
      //支持linux,将所有\换成/
      dir[i] = '/';
    }
  }

  return 0;
}

int txt2mnist_x(string& filename, const unsigned images, const unsigned rows,
    const unsigned cols)
{
  ifstream is;
  string in_name = dir + filename + ".txt";
  is.open(in_name, ios::in);
  if (!is) {
    cerr << "open file error: " << in_name << endl;
    return -1;
  }
  unsigned char buf[16];
  buf[0] = buf[1] = 0;
  if (typeid(out_t) == typeid(unsigned char))
    buf[2] = 0x8;
  else if (typeid(out_t) == typeid(char))
    buf[2] = 0x9;
  else if (typeid(out_t) == typeid(int16_t))
    buf[2] = 0xB;
  else if (typeid(out_t) == typeid(int32_t))
    buf[2] = 0xC;
  else if (typeid(out_t) == typeid(float))
    buf[2] = 0xD;
  else if (typeid(out_t) == typeid(double)) {
    buf[2] = 0xE;
  } else {
    cerr << typeid(out_t).name() << endl;
    return -1;
  }
  buf[3] = 3; // dim
  char* sbuf = (char*)buf;
  unsigned *ibuf = (unsigned*)buf, magic = ntohl(ibuf[0]), tt = buf[2],
           dim = buf[3];
  cout << magic << '\t' << tt << '\t' << dim << '\t' << images << '\t' << rows
       << '\t' << cols << endl;
  unsigned bsize = images * rows * cols;
  vector<out_t> data(bsize);
  for (unsigned i = 0, p = 0, v; i < images; ++i) {
    for (unsigned r = 0; r < rows; ++r) {
      for (unsigned c = 0; c < cols; ++c) {
        is >> v;
        data[p++] = v;
      }
    }
  }
  cout << "read file: " << in_name << " " << data[0] << endl;
  ibuf[1] = htonl(images);
  ibuf[2] = htonl(rows);
  ibuf[3] = htonl(cols);
  string out_name = dir + filename + ".dat";
  ofstream os;
  os.open(out_name, ios::binary | ios::out);
  if (!os) {
    cerr << "open file error: " << out_name << endl;
    return -1;
  }
  os.write(sbuf, 16);
  os.write((char*)&data[0], bsize * sizeof(out_t));
  os.close();
  cout << "save to file: " << out_name << endl;
  cout << ntohl(ibuf[0]) << "\t" << (unsigned)buf[2] << "\t"
       << (unsigned)buf[3] << "\t" << images << "\t" << rows << "\t" << cols
       << endl;
  return 0;
}

int txt2mnist_y(string& filename, const unsigned images)
{
  ifstream is;
  string in_name = dir + filename + ".txt";
  is.open(in_name, ios::in);
  if (!is) {
    cerr << "open file error: " << in_name << endl;
    return -1;
  }
  unsigned char buf[8];
  buf[0] = buf[1] = 0;
  if (typeid(out_t) == typeid(unsigned char))
    buf[2] = 0x8;
  else if (typeid(out_t) == typeid(char))
    buf[2] = 0x9;
  else if (typeid(out_t) == typeid(int16_t))
    buf[2] = 0xB;
  else if (typeid(out_t) == typeid(int32_t))
    buf[2] = 0xC;
  else if (typeid(out_t) == typeid(float))
    buf[2] = 0xD;
  else if (typeid(out_t) == typeid(double)) {
    buf[2] = 0xE;
  } else {
    cerr << typeid(out_t).name() << endl;
    return -1;
  }
  buf[3] = 1; // dim
  char* sbuf = (char*)buf;
  unsigned *ibuf = (unsigned*)buf, magic = ntohl(ibuf[0]), tt = buf[2],
           dim = buf[3];
  cout << magic << '\t' << tt << '\t' << dim << '\t' << images << endl;
  vector<out_t> data(images);
  for (unsigned i = 0, v; i < images; ++i) {
    is >> v;
    data[i] = v;
  }
  cout << "read file: " << in_name << endl;
  ibuf[1] = htonl(images);
  string out_name = dir + filename + ".dat";
  ofstream os;
  os.open(out_name, ios::binary | ios::out);
  if (!os) {
    cerr << "open file error: " << out_name << endl;
    return -1;
  }
  os.write(sbuf, 8);
  os.write((char*)&data[0], images * sizeof(out_t));
  os.close();
  cout << "save to file: " << out_name << endl;
  cout << ntohl(ibuf[0]) << "\t" << (unsigned)buf[2] << "\t"
       << (unsigned)buf[3] << "\t" << images << endl;
  return 0;
}

//*****************************************************************//*
int main(int argc, char** argv)
{
  if (argc > 1) {
    for (int i = 1; i < argc; ++i)
      cout << '\t' << argv[i];
    cout << endl;
  }
  mkdirs(outdir.c_str());

#if DIGITS_TXT2MNIST
  txt2mnist_x(filenames[0], images, rows, cols);
  txt2mnist_y(filenames[1], images);
#elif DIGITS_COUNT
  for (auto& filename : filenamesx) {
    mnist_x(filename);
  }
  for (auto& filename : filenamesy) {
    mnist_y(filename);
  }
#else // MNIST_COUNT

  for (auto& filename : filenamesx) {
    mnist_x(filename);
  }
  for (auto& filename : filenamesy) {
    mnist_y(filename);
  }
#endif

  return 0;
}

//*****************************************************************//*
#endif // _MAIN_
