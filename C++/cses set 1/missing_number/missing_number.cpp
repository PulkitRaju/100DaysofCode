#include <bits/stdc++.h>
#include <iostream>
using namespace std;
#define TxtIO   freopen("input.txt","r",stdin); freopen("output.txt","w",stdout);
int main(){
    TxtIO;
    long long n;
    // vector<long long> g1;
    // n=3;
    cin>>n;
    long long expected=(n*(n+1))/2;
    long long actual=0;
    for(long long i=0; i<n-1;i++){
        long long a;
        cin>>a;
        actual=actual+a;

    }
    
    long long missing=expected-actual;
    cout<<missing;
    // g1.push_back(n);
    
}