#include <bits/stdc++.h>
#include <random>
using namespace std;

void rand_int_func(std::function<int(void)> const& random_int_generate)
{
    int ret = random_int_generate();
    cout << "ret is: " << ret << endl;
    return;
}

void take_gen_dist(default_random_engine &e, uniform_int_distribution<int> &d)
{
    int roll = d(e);
    cout << "roll is: " << roll << endl;
    return;
}


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
    // invoke function to take in bound function
    cout << "Calling rand_int_func(dice)" << endl;
    rand_int_func(dice); 
    for(int i=0; i<10; ++i)
    {
        rand_int_func(dice);
    }
    // invoke function to take in genrator and distribution
    cout << "Calling take_gen_dist(gen, dis)" << endl;
    for(int i=0; i<20; ++i)
    {
        take_gen_dist(generator, distribution);
    }

    return 0;
}
