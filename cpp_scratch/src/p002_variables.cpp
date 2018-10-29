#include <iostream>

using namespace std;

int main()
{
    // Initialize x
    int x=5;

    // Assign x some value
    x = x - 2;
    cout<<x<<endl;

    int y = x;
    cout<<y<<endl;

    // x+y is a r-value
    cout<<x+y<<endl;
    cout<<x<<endl;

    // Uninitialized variable
    int z;
    cout<<z<<endl;

    return 0;
}
