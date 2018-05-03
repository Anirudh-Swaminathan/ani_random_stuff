#include <iostream>

// The following line works only if the program is in the same project
// int add(int x,  int y);

// The following is a try of my hack for making it work
#include "p008_add.h"

int main()
{
    std::cout << "The sum of 3 and 4 is: " << add(3, 4) << std::endl;
    return 0;
}
