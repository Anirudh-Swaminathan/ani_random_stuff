#include <iostream>
#include <unordered_map>
using namespace std;

class RomanConverter
{
private:
    string inp;
    int out;

    unordered_map<string, int> ma = {
            {"I", 1},
            {"IV", 4},
            {"V", 5},
            {"IX", 9},
            {"X", 10},
            {"XL", 40},
            {"L", 50},
            {"XC", 90},
            {"C", 100},
            {"CD", 400},
            {"D", 500},
            {"CM", 900},
            {"M", 1000}
        };
public:

    RomanConverter()
    {
        out = -1;
        inp = "";
    }

    void take_in()
    {
        cout << "\nPlease enter a valid Roman Numeral: ";
        cin >> inp;
    }
    void convert()
    {
        //TODO: Implement the conversion
        //cout << "\nYet to be implemented";
        string cur = "";
        string prev = "";
        int ans = 0;
        for(int i=inp.length()-1; i>=0; --i)
        {
            prev = cur;
            cur = inp[i] + cur;
            //cout<<"\n"<<i<<" "<<prev<<" "<<cur<<" "<<ma[cur];
            if(!ma[cur])
            {
                ans += ma[prev];
                cur = inp[i];
                prev = "";
            }
        }
        ans += ma[cur];
        out = ans;
    }
    void print_out()
    {
        //cout << "\n" << ma[inp];
        cout << "\nThe integer corresponding to " << inp << " is: " << out;
    }
};

int main()
{
    RomanConverter rc;
    rc.take_in();
    rc.convert();
    rc.print_out();
    return 0;
}
