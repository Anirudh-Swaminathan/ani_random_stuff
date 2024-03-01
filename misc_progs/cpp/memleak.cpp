#include <iostream>
#include <memory>
 
struct Foo
{
    Foo(int n = 0) noexcept : bar(n)
    {
        std::cout << "Foo::Foo(), bar = " << bar << " @ " << this << '\n';
    }
    ~Foo()
    {
        std::cout << "Foo::~Foo(), bar = " << bar << " @ " << this << '\n';
    }
    int getBar() const noexcept { return bar; }
private:
    int bar;
};
 
int main()
{
    void* ptr = new Foo(42);
    ((Foo*)ptr)->getBar();
    // comment out next line to trigger a memory leak
    delete ((Foo*)ptr);
}
