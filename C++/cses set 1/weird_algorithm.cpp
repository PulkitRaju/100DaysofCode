#include <bits/stdc++.h>
#include <iostream>
using namespace std;

int main(){
    long long n;
    vector<long long> g1;
    // n=3;
    cin>>n;
    g1.push_back(n);
    
    while(n!=1){
        if (n%2==0){
            n=n/2;
            g1.push_back(n);
        }
        else{
            n=n*3+1;
            g1.push_back(n);
        }
        
    }
    
    for (auto it = g1.begin(); it != g1.end(); it++) 
        cout << *it << " "; 
}
