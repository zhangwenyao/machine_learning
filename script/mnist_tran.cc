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

using out_t = float;
// auto& tran = tran8;
// string dir = "../data/MNIST_data/", outdir = "../data/MNIST_count8_data/";
auto tran = tran_func80n4;
string dir = "../data/MNIST_data/", outdir = "../data/MNIST_count80n4_data/";
string filenamesx[2]
    = { "t10k-images-idx3-ubyte", "train-images-idx3-ubyte" },
    filenamesy[2] = { "t10k-labels-idx1-ubyte", "train-labels-idx1-ubyte" };
constexpr unsigned rows = 28, cols = 28, len = 4 * (rows + cols);
double ratio = 1.0;

int handlex(string& filename)
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
  vector<out_t> res(outlen, 0);
  if (0 != tran8(&data[0], &res[0], images, rows, cols, tran)) {
    return -1;
  }
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

int handley(string& filename)
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
//*****************************************************************//*
int main(int argc, char** argv)
{
  if (argc > 1) {
    for (int i = 1; i < argc; ++i)
      cout << '\t' << argv[i];
    cout << endl;
  }
  mkdirs(outdir.c_str());
  for (auto& filename : filenamesx) {
    handlex(filename);
  }
  for (auto& filename : filenamesy) {
    handley(filename);
  }
  return 0;
}

//*****************************************************************//*
#endif // _MAIN_
