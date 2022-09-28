// #include <bits/stdc++.h>
#include <iostream>
#include <map>
#include <string>
#include <fstream>
#include <algorithm>
#include <vector>
#include <set>

using namespace std;
set<string> names;

void fill_map(ifstream& fin,map<string,double>& m)
{
  {
    string s;
    getline(fin,s);
    getline(fin,s);
  }
  while(!fin.eof())
  {
    vector<string> v(5);
    for(int i=0;i<5;++i)
    {
      fin>>v[i];
    }
    // cout<<v[0]<<" "<<v.back()<<"\n";
    if(m.find(v[0])==m.end())
    {
      m[v[0]]=stod(v.back());
    }
    else
    {
      m[v[0]]=min(m[v[0]],stod(v.back()));
    }
    names.insert(v[0]);
  }

}

int main()
{
  std::map<std::string,double> woa,sso,pso,drag;
  ifstream f_woa("woa.txt",ios_base::in);
  ifstream f_sso("sso.txt",ios_base::in);
  ifstream f_pso("pso.txt",ios_base::in);
  ifstream f_drag("dragonfly.txt",ios_base::in);

  fill_map(f_woa,woa);
  fill_map(f_sso,sso);
  fill_map(f_pso,pso);
  fill_map(f_drag,drag);
  cout.precision(10);
  for(auto &x: names)
  {
    cout<<"\\hline"<<"\n";
    cout<<x<<" & "<<woa[x]<<" & "<<sso[x]<<" & "<<pso[x]<<" & "<<drag[x]<<"\\\\\n";
  }
}