#include <iostream>

// function to return twice the value of an integer
// this function satisfies the 4th question of the quiz
int doubleNumber(int x)
{
    return 2*x;
}

int userInput()
{
    int x;
    std::cout << "Please enter an integer; ";
    std::cin >> x;
    return x;
}

int main()
{
    int s = userInput();
    int res = doubleNumber(s);
    std::cout << "2 * " << s << " = "  << res << std::endl;
}
