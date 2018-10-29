#include <iostream>
#include "p013_square.h"

int main()
{
    std::cout << "A square has " << getSquareSides() << " sides " << std::endl;
    std::cout << "A square of length 5 has perimeter " << getSquarePerimeter(5) << std::endl;
    return 0;
}
