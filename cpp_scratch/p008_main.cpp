#include <iostream>

// The following line works only if the program is in the same project
// int add(int x,  int y);

// This is a forward declaration of add() that is linked at run-time
// when compiled along with p008_add.cpp
// int add(int, int);

// The following line includes a user-defined header file
#include "p008_add.h"

int main()
{
    std::cout << "The sum of 3 and 4 is: " << add(3, 4) << std::endl;
    return 0;
}
