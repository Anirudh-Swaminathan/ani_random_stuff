// Program to add two integers with functions defined in separate files

#include<iostream>

int readNumber();
void writeAnswer(int);

int main()
{
    int a = readNumber();
    int b = readNumber();
    writeAnswer(a + b);
    return 0;
}
