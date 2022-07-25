#include <bits/stdc++.h>
#include <random>
using namespace std;

int main()
{
    // initialize the random generator
    default_random_engine generator;
    // define the distribution
    // here, it is a uniform int distribution in our desired range
    uniform_int_distribution<int> distribution(1, 6);
    int dice_roll = distribution(generator);
    cout << "dice_roll: " << dice_roll << endl;
    for (int i=0; i<10; ++i)
    {
        dice_roll = distribution(generator);
        cout << "dice_roll: " << dice_roll << endl;
    }
    // use bind instead
    auto dice = std::bind(distribution, generator);
    int w = dice() + dice() + dice();
    cout << "triple w: " << w << endl;
    for (int i=0; i<10; ++i)
    {
        w = dice();
        cout << "w: " << w << endl;
    }

    return 0;
}
