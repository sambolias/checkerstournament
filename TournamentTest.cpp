#include "Tournament.h"
#include <vector>
using std::vector;
#include <iostream>
using std::cout;

int main()
{
    int maxMoves = 200;
    int populationSize = 60;
    double winWeight = 1.0;
    double lossWeight = -2.0;
    double drawWeight = 0.1;
    vector<int> layers = {32, 96, 40, 16, 1};
    Tournament t;
cout<<"hi\n";
    t.Start(populationSize, maxMoves, winWeight, lossWeight, drawWeight, layers);
}
