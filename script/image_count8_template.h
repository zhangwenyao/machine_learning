#ifndef IMAGE_COUNT8_TEMPLATE_H_
#define IMAGE_COUNT8_TEMPLATE_H_

template <typename T, typename T2>
int tran8(unsigned char* source, T* target, const unsigned images,
    const unsigned rows, const unsigned cols, T2 func)
{
  for (unsigned i = 0; i < images; ++i) {
    func(
        source + i * rows * cols, target + i * 4 * (rows + cols), rows, cols);
  }
  return 0;
}

#endif
