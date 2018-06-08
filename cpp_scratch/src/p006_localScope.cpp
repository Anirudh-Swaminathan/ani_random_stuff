#include <iostream>

int add(int x, int y)
{
    return x+y;
}

int main()
{
    int x = 5;
    int y = 6;
    std::cout << add(x, y) << std::endl;
    return 0;
}
