#include <bits/stdc++.h>
#include <iostream>
using namespace std;
#define TxtIO   freopen("input.txt","r",stdin); freopen("output.txt","w",stdout);
int main(){
    TxtIO;
    string n;
    // vector<long long> lcq;
    cin>>n;
    
    
    long long curr_sum=1;
    long long max_sum=1;
    long long len=n.length();
    for(long long i=1;i<=len;i++){
        if(n[i]==n[i-1]){
            curr_sum=curr_sum+1;
            max_sum=max(curr_sum,max_sum);

        }
        else{
            
            curr_sum=1;
        }

    }
    cout<<max_sum;
    

  
    // g1.push_back(n);
    
}