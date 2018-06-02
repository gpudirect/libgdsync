#include "wq.hpp"
#include <unistd.h>

bool simple_fun()
{
    static int j = 0;
    printf("%s j=%d\n", __FUNCTION__, j);
    return ++j < 1000 ? true : false;
}

struct functor {
    functor() : k(0) {}

    bool run() {
        printf("%s k=%d\n", __FUNCTION__, k);
        return ++k < 200 ? true : false;
    }
    int k;
};

main()
{
    work_queue wq;
    wq.queue(&simple_fun);
    functor f;
    wq.queue(std::bind(&functor::run, &f));
    sleep(1);
}


// compile: c++ -g -std=c++0x wq.cpp -o wq
