#include<iostream>

int main()
{
    unsigned short x = 65535;
    std::cout << "x was " << x << std::endl;
    x = x + 4;
    std::cout << "x is now " << x << std::endl;
    unsigned short p = 0;
    std::cout << "p was " << p << std::endl;
    p = p - 1;
    std::cout << "p is now " << p << std::endl;
    signed short i = 32767;
    std::cout << "i was " << i << std::endl;
    i = i + 1;
    std::cout << "i is now " << i << std::endl;
    signed short k = -32768;
    std::cout << "k was " << k << std::endl;
    k = k - 1;
    std::cout << "k is now " << k << std::endl;
    return 0;
}
