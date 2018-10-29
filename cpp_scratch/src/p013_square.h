#ifndef SQUARE_H
#define SQUARE_H

// Function definition to return the number of sides of a square
/*
 * This definition in the header causes problems when both square.cpp
 * and main.cpp are compiled together. This is because, this definition
 * is included in both source files, and hence causes a linker error.
 * To prevent this, we remove the definition of the files from the header
 * Another is to use #pragma once
*/
int getSquareSides()
{
    return 4;
}

// Foward declaration for calculating the perimeter of a square
int getSquarePerimeter(int sideLength);
#endif
