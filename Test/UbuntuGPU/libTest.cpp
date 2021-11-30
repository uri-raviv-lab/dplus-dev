#include <iostream>
#include <dlfcn.h>


int main()
{
  void *ptrHandle = dlopen("./libgpubackend.so", RTLD_NOW);
  if(ptrHandle)
    std::cout << "Success!!\n";
  else
    std::cout << ":(\n";

  std::cout << dlerror() << "\n";
  return 0;
}

