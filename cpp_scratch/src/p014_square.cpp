#include "p014_square.h"

// Defining the getSquareSides() function here
int getSquareSides()
{
    return 4;
}

int getSquarePerimeter(int sideLength)
{
    return sideLength*getSquareSides();
}
